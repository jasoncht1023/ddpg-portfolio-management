import matplotlib.pyplot as plt
import numpy as np
import csv
import os

# sharpe ratio of uniformly weighted portfolio
def uniformWeightingSR(dateAxis, adjCloseMatrix, riskFreeRates):
    monthlyReturnDiffUniWeight = dict()
    prevPortValue = np.average(adjCloseMatrix[0])
    for i in range(len(dateAxis)-1):
        if (dateAxis[i][5:7] != dateAxis[i+1][5:7]):
            monthlyReturn = (np.average(adjCloseMatrix[i]) - prevPortValue) / prevPortValue
            monthlyReturnAnnualized = (1 + monthlyReturn) ** 12 - 1
            print(dateAxis[i][:7], "val:", np.average(adjCloseMatrix[i]), "prev:", prevPortValue, "annualized:", monthlyReturnAnnualized)
            monthlyReturnDiffUniWeight[dateAxis[i][:7]] = monthlyReturnAnnualized - riskFreeRates[dateAxis[i][:7]] / 100
            prevPortValue = np.average(adjCloseMatrix[i])
    print("Sharpe Ratio for uniform portfolio weighting:", np.average(list(monthlyReturnDiffUniWeight.values())) / np.std(list(monthlyReturnDiffUniWeight.values())))

# portfolio assets: MSFT AAPL GOOG MMM GS NKE AXP HON CRM JPM
def main():
    adjCloseMatrix = list()
    riskFreeRates = dict()
    dateAxis = np.array([])
    MA = list()
    MA_window = 28

    # load the adjusted close price data for the assets
    with open('stock.csv', mode ='r') as file:
        csvFile = list(csv.reader(file))
        # adjCloseMatrix.append(["Date"] + csvFile[1][1:11])
        for i in range(3, len(csvFile)):
            dateAxis = np.append(dateAxis, csvFile[i][0][:10])
            adjCloseMatrix.append(list(map(float, csvFile[i][1:11])))
        adjCloseMatrix = np.array(adjCloseMatrix)

    # load the risk free rate data
    with open('30y-treasury-rate.csv', mode ='r') as file:
        csvFile = list(csv.reader(file))
        for i in range(1, len(csvFile)-1):
            date = list(csvFile[i][0].split("-"))
            nextDate = list(csvFile[i+1][0].split("-"))
            if (date[1] != nextDate[1]):
                riskFreeRates[csvFile[i][0][:7]] = float(csvFile[i][1])
        riskFreeRates[csvFile[len(csvFile)-1][0][:7]] = float(csvFile[len(csvFile)-1][1])

    print(dateAxis)
    print(adjCloseMatrix)

    uniformWeightingSR(dateAxis, adjCloseMatrix, riskFreeRates)
    # uniWeightPortfolioValue = [np.average(column) for column in adjCloseMatrix]

    # Computing the Moving Average matrix
    # for i in range(len(dateAxis)):
    #     dayMA = list()
    #     for j in range(len(adjCloseMatrix[0])):
    #         dayMA.append(np.average([column[j] for column in adjCloseMatrix[max(0, i-MA_window+1):i+1]]))
    #     MA.append(dayMA)
    # MA = np.array(MA)

    # if os.path.exists("output.txt"):
    #     os.remove("output.txt")
    # f = open("output.txt", "a")
    # for i in range(len(dateAxis)):
    #     f.write(dateAxis[i] + " " + str(MA[i]) + "\n")
    # f.close()

    # print(MA)

    # plt.plot(dateAxis, uniWeightPortfolioValue)
    # plt.show()

main()