from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt

def main():
    #create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt(open('Data/train.csv','r'), delimiter=',', dtype='f8')[1:]    
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    test = genfromtxt(open('Data/test.csv','r'), delimiter=',', dtype='f8')[1:]

    #create and train the random forest
    #rf = RandomForestClassifier(n_estimators=100)
    #use 3 of my CPU cores
    rf = RandomForestClassifier(n_estimators=100, n_jobs=3)
    rf.fit(train, target)
    predicted_probs = [x[1] for x in rf.predict_proba(test)]

    savetxt('Data/submission.csv', predicted_probs, delimiter=',', fmt='%f')

if __name__=="__main__":
    main()
