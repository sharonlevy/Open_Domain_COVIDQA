import os
import json
import sys
import datetime
import ipdb

# check if our date is in range
def dateInRange(startDay, endDay, docDate, monthOnly,month, yearOnly,year):
    if monthOnly:
        startMonth = int(startDay.strftime("%m"))
        startYear = int(startDay.strftime("%Y"))
        endMonth = int(endDay.strftime("%m"))
        endYear = int(endDay.strftime("%Y"))
        if year > startYear:
            if year < endYear:
                return True
            elif year == endYear and month <= endMonth:
                return True
            else:
                return False
        elif year == startYear:
            if month >= startMonth:
                if year < endYear:
                    return True
                elif year == endYear and month <= endMonth:
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False
    elif yearOnly:
        startYear = int(startDay.strftime("%Y"))
        endYear = int(endDay.strftime("%Y"))
        if year >= startYear and year <= endYear:
            return True
    elif startDay <= docDate and docDate <= endDay:
        return True
    else:
        return False
    return False

def checkDate(startDay, endDay, docDate):
    dateStr = docDate
    dateArr = []
    # first case
    if "-" in dateStr:
        dateArr = dateStr.split('-')
    # else, it's the other 3 cases
    else:
        dateArr = dateStr.split()
        # second case (and fill in the month for third case)
        if len(dateArr) == 3 or len(dateArr) == 2:
            count = 1
            for month in months:
                if dateArr[1] == month:
                    dateArr[1] = str(count) if count >= 10 else "0" + str(count)
                    break
                count += 1
        # now: fix the day (this is for third and fourth case)
        # third case
        if len(dateArr) == 2:
            count = 0
            # we MUST change all the strings in array to numbers
            for i, date in enumerate(dateArr):
                dateArr[i] = int(date)
            # two dates: first day of month, and last day of month (checking
            # if this date could possibly be in range)

            # is it in range?
            if dateInRange(startDay, endDay, startDay, True, int(dateArr[1]), False, int(dateArr[0])):
                return True
            else:
                return False
        elif len(dateArr) == 1:
            # two dates: jan 1st and december 31st (checking
            # if this date could possibly be in range)

            if dateInRange(startDay, endDay, startDay, False, -1, True, int(dateArr[0])):
                return True
            else:
                return False
    # first and second case, we still don't have answer
    # convert all to integers
    for i, date in enumerate(dateArr):
        dateArr[i] = int(date)
    currentDate = datetime.datetime(dateArr[0], dateArr[1], dateArr[2])
    if dateInRange(startDay, endDay, currentDate, False, -1, False, -1):
        return True
    else:
        return False