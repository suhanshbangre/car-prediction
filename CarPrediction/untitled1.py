var = 5
def increase():
    global var
    if var>0:
        var+=1
        print(var)
    
increase()
increase()
increase()
increase()