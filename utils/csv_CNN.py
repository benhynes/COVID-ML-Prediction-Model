import csv
import os

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
        filename = os.path.join(dirname, '../frontend/forecast/p{}.csv'.format(date))
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