def strangeCounter(t):
    tCount=3

    timeleast=0
    while(timeleast+tCount<t):
        timeleast+=tCount
        tCount=tCount*2
    print(t)
    print(timeleast)
    print(tCount)
    return  tCount-(t-timeleast)+1

print(strangeCounter(4))