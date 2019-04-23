import tree, csv

if __name__ == "__main__":
    reader = csv.reader('./data/training_set.csv')
    training_obs = []
    training_cat = []
    for line in reader:
        training_obs.append(line[:-1])
        training_cat.append(line[-1])
    tree.train(training_obs, training_cat, "tree.xml")
    
    reader = csv.reader('./data/training_set.csv')
    answer = []
    testing_obs = []
    for line in reader:
        testing_obs.append(line[:-1])
        answer.append(line[-1])
    answer.pop(0)
    
    prediction = tree.predict("DecisionTree.xml",testing_obs)
    err = 0
    for i in range(len(answer)):
        if not answer[i]==prediction[i]:
            err=err+1
    print("error rate=", round(float(err)/len(prediction)*100,2), "%")
    
        