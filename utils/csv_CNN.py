import csv
import os

def cleanData():
    dirname = os.path.dirname(__file__)
    directory = os.path.join(dirname, '../frontend/forecast')
    forecast_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    for file in forecast_files:
        file_dir = os.path.join(dirname, '../frontend/forecast/{}'.format(file))
        os.delete(file_dir)

    directory = os.path.join(dirname, '../frontend/past_data')
    past_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    for file in past_files:
        file_dir = os.path.join(dirname, '../frontend/past_data/{}'.format(file))
        os.delete(file_dir)

def parseToCSV(ans):
    dirname = os.path.dirname(__file__)
    for date in range(len(ans)):
        filename = os.path.join(dirname, '../frontend/forecast/f{}.csv'.format(date))
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            
            rowTemp = []
            
            rowSize = ans[date].shape[0]
            colSize = ans[date].shape[1]
            
            for row in range(rowSize):
                for item in range(colSize):
                    rowTemp.append(ans[date][row][item])
                writer.writerow(rowTemp)
                rowTemp = []
                
def parsePastToCSV(past):
    dirname = os.path.dirname(__file__)
    for date in range(len(past), 1):
        filename = os.path.join(dirname, '../frontend/past_data/p{}.csv'.format(date))
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            
            rowTemp = []
            
            rowSize = past[date].shape[0]
            colSize = past[date].shape[1]
            
            for row in range(rowSize):
                for item in range(colSize):
                    rowTemp.append(past[date][row][item])
                writer.writerow(rowTemp)
                rowTemp = []