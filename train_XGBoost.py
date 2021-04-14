from utils.datalib import *
import copy

if __name__ == "__main__":

    urls = ['https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv', 
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv',
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv']

    confirmed_raw_dataset = read_csv(urls[0])
    deaths_raw_dataset = read_csv(urls[1])
    recovered_raw_dataset = read_csv(urls[2])

    confirmed_dataset = preprocess(confirmed_raw_dataset)
    deaths_dataset = preprocess(deaths_raw_dataset)
    recovered_dataset = preprocess(recovered_raw_dataset)

    coordinates = extract_coordinates(recovered_raw_dataset)

    c_map = get_loc_map(coordinates, confirmed_dataset)
    r_map = get_loc_map(coordinates, recovered_dataset)
    d_map = get_loc_map(coordinates, deaths_dataset)

    maps_dict = {
        'c_map': c_map,
        'r_map': r_map,
        'd_map': d_map
    }

    print("c_map shape: ", c_map.shape)
    print("r_map shape: ", r_map.shape)
    print("d_map shape: ", d_map.shape)

    X = np.zeros((3, len(maps_dict['c_map'].flatten())))
    y = []

    X[0] = maps_dict['c_map'].flatten() 
    X[1] = maps_dict['r_map'].flatten() 
    X[2] = maps_dict['d_map'].flatten() 

    y = copy.deepcopy(X)
    X = X.transpose()
    y = y.transpose()

    dim1, dim2 = X.shape

    y = y.flatten()

    l = y.shape

    y = y[l[0] - dim1:]

    print("X shape: ", X.shape)
    print("y shape: ", y.shape)

    # dividing data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.5)
    # instantiating regression model


    

    print("X_train shape: ", X_train.shape)
    print("X_valid shape: ", X_valid.shape)
    print("y_train shape: ", y_train.shape)
    print("y_valid shape: ", y_valid.shape)


    try:
        # code for loading the model


        loaded_model = xgb.Booster()
        

        prediction = loaded_model.predict(X_full)



    except IOError:

        covid_model = XGBRegressor(n_estimators=500, learning_rate=0.05, n_jobs=4)

        # fitting regression model with early_stopping_rounds, which tests model fit with increasing
        # n_estimators and stops when there are 5 consecutive increases in MAE
        covid_model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_valid, y_valid)], verbose=False)

        print("X_valid", X_valid)
        print(type(X_valid))
        #X_valid.sort_index(inplace=True)
        print("X_valid_sorted", X_valid)
        print("X", X)

        # outputs prediction values for the set
        predictions = covid_model.predict(X_valid)

        # generates MAE value
        print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))

        # determining baseline for MAE 
        mean_train = np.mean(y_train)

        # gets predictions on the test set
        baseline_predictions = np.ones(y_valid.shape) * mean_train

        # computes MAE baseline
        mae_baseline = mean_absolute_error(y_valid, baseline_predictions)

        print("Baseline MAE is {:.2f}".format(mae_baseline))

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

        print(optimized_covid_model_dm.predict(dtest))

        optimized_covid_model_dm.save_model("xgboost_covid_optimized.model")

        plot_tree(optimized_covid_model_dm)

        plt.show()

