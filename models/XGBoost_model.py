
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

class XGBoost_Model():
    def __get_model(self, n_estimators, lr, n_jobs):
        model = XGBRegressor(n_estimators = n_estimators, learning_rate = lr, n_jobs = n_jobs)
        return model

    def __init__(self, name, n_estimators = 500, learning_rate = 0.05, n_jobs = 4):
        self.name = name
        self.model = self.get_model(n_estimators, learning_rate, n_jobs)
    
    def predict(self, x):
        return self.model.predict(x)

    def fit(self, x, y):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_valid, label=y_valid)

        self.model.fit(x, y, early_stopping_rounds=5, verbose=False)

        
        # managing parameters for model
        params = {
            # Parameters that we are going to tune.
            'max_depth':11,
            'min_child_weight': 1,
            'eta':.01,
            'subsample': 1.0,
            'colsample_bytree': 0.9,
            # Other parameters
            'objective':'reg:squarederror',
            'eval_metric': "mae",
        }

        # max number
        num_boost_round = 999

        # secondary covid model using dmatrix
        covid_model_dm = xgb.train(
            params, 
            dtrain, 
            num_boost_round=num_boost_round, 
            evals=[(dtest, "Test")], 
            early_stopping_rounds=10
        )

        # secondary covid results with different approach
        dm_cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics={'mae'},
            early_stopping_rounds=10
        )

        print(dm_cv_results['test-mae-mean'].min())

        # these are ranges that can be adjusted

        gridsearch_params = [
            (max_depth, min_child_weight)
            for max_depth in range(9,12)
            for min_child_weight in range(1,8)
        ]


        # not necessaryto run cross validation each time
        # best params found are max_depth=11, min_child_weight=1

        # define initial best params and MAE
        min_mae = float("Inf")
        best_params = None
        for max_depth, min_child_weight in gridsearch_params:
            print("CV with max_depth={}, min_child_weight={}".format(
                                    max_depth,
                                    min_child_weight))

            # Update our parameters
            params['max_depth'] = max_depth
            params['min_child_weight'] = min_child_weight

            # Run CV
            cv_results = xgb.cv(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                seed=42,
                nfold=5,
                metrics={'mae'},
                early_stopping_rounds=10
            )

            # Update best MAE
            mean_mae = cv_results['test-mae-mean'].min()
            boost_rounds = cv_results['test-mae-mean'].argmin()
            print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
            if mean_mae < min_mae:
                min_mae = mean_mae
                best_params = (max_depth,min_child_weight)

        print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))


        # best settings as output by tests above
        params['max_depth']=best_params[0]
        params['min_child_weight']=best_params[1]

        # best params subsample=1.0, colsample_bytree=0.9

        gridsearch_params = [
            (subsample, colsample)
            for subsample in [i/10. for i in range(7,11)]
            for colsample in [i/10. for i in range(7,11)]
        ]

        min_mae = float("Inf")
        best_params = None
        # We start by the largest values and go down to the smallest
        for subsample, colsample in reversed(gridsearch_params):
            print("CV with subsample={}, colsample={}".format(
                                    subsample,
                                    colsample))
            # We update our parameters
            params['subsample'] = subsample
            params['colsample_bytree'] = colsample
            # Run CV
            cv_results = xgb.cv(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                seed=42,
                nfold=5,
                metrics={'mae'},
                early_stopping_rounds=10
            )
            # Update best score
            mean_mae = cv_results['test-mae-mean'].min()
            boost_rounds = cv_results['test-mae-mean'].argmin()
            print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
            if mean_mae < min_mae:
                min_mae = mean_mae
                best_params = (subsample,colsample)

        print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))



        # best settings as output by tests above
        params['subsample']=best_params[0]
        params['colsample_bytree']=best_params[1]

        # eta=0.01

        # This can take some time…
        min_mae = float("Inf")
        best_params = None
        for eta in [.3, .2, .1, .05, .01, .005]:
            print("CV with eta={}".format(eta))
            # We update our parameters
            params['eta'] = eta
            # Run and time CV
            cv_results = xgb.cv(
                    params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    seed=42,
                    nfold=5,
                    metrics=['mae'],
                    early_stopping_rounds=10
                )
            # Update best score
            mean_mae = cv_results['test-mae-mean'].min()
            boost_rounds = cv_results['test-mae-mean'].argmin()
            print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
            if mean_mae < min_mae:
                min_mae = mean_mae
                best_params = eta
        print("Best params: {}, MAE: {}".format(best_params, min_mae))

        # best settings as output by tests above
        params['eta']=best_params

        num_boost_round = covid_model_dm.best_iteration + 1

        optimized_covid_model_dm = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtest, "Test")]
        )

        print(mean_absolute_error(optimized_covid_model_dm.predict(dtest), y_valid))

    def train_on_batch(self, x_batch, y_batch):
        return self.model.train_on_batch(x_batch,y_batch)

    def save_weights(self, file_name = "FC_Model.h5"):
        self.model.save_weights(file_name)
    
    def load_model(self, file_name = "XGBoost_model.h5"):
        self.model = xgb.Booster()
        self.model.load_model("xgboost_covid_optimized.model")
    

