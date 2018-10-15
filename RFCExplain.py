import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
import json
import operator
import os
import shutil
import time
import numpy as np
import graphviz
from sklearn import tree
import pickle
import sys

def printPathAttributesIntoFile(estimator,X_test,sample_id,filename,headers):
	node_indicator = estimator.decision_path(X_test)
	leave_id = estimator.apply(X_test)
	node_index = node_indicator.indices[node_indicator.indptr[sample_id]:node_indicator.indptr[sample_id + 1]]
	n_nodes = estimator.tree_.node_count
	children_left = estimator.tree_.children_left
	children_right = estimator.tree_.children_right
	feature = estimator.tree_.feature
	threshold = estimator.tree_.threshold

	with open(filename,"w") as f:
		s='Rules used to predict sample %s: \n' % sample_id
		f.write(s)
		for node_id in node_index:
			#f.write("Test 1")
			if leave_id[sample_id] == node_id:
				continue
			if (float(X_test.iloc[sample_id,feature[node_id]]) <= threshold[node_id]):
				threshold_sign = "<="
			else:
				threshold_sign = ">"
			s="decision id node %s : (X_test[%s, %s] (= %s) %s %s)\n"% (node_id,sample_id,headers[feature[node_id]],X_test.iloc[sample_id,feature[node_id]],threshold_sign,threshold[node_id])
			f.write(s)
		output=estimator.predict([X_test.iloc[sample_id]])
		s="Predicted Output is :"+str(output[0])
		f.write(s)
	f.close()

def printPathsIntoFile(testX,model,projectName,headers):
	print("Starting Paths Printing on test data.")

	toolbarWidth=40
	print("Progress ->",end=" ")
	sys.stdout.write("[%s]" % (" " * toolbarWidth))
	sys.stdout.flush()
	sys.stdout.write("\b" * (toolbarWidth+1))
	
	#print(testY[0])
	l=len(te)
	mult=1
	prod=l//toolbarWidth
	createSubDirectory(projectName,"Paths")
	subDirName=projectName+"/Paths/"
	for j in range(len(testX)):
		i=1
		sampleNumber=subDirName+"/Sample#"+str(j+1)
		folderName="Sample#"+str(j+1)
		createSubDirectory(subDirName,folderName)
		for estimator in model.estimators_:
			treeName="TreeNumber#"+str(i)+".txt"
			filename=sampleNumber+"/Tree#"+str(i)+".txt"
			printPathAttributesIntoFile(estimator,testX,j,filename,headers)
			i+=1
			if(j==prod*mult):
				sys.stdout.write("|")
				sys.stdout.flush()
				mult+=1

def printTreeStructure(estimator,headers,treeNumber,projectName):
	n_nodes = estimator.tree_.node_count
	children_left = estimator.tree_.children_left
	children_right = estimator.tree_.children_right
	feature = estimator.tree_.feature
	threshold = estimator.tree_.threshold

	output=""
	node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
	is_leaves = np.zeros(shape=n_nodes, dtype=bool)
	stack = [(0, -1)]  # seed is the root node id and its parent depth
	while len(stack) > 0:
		node_id, parent_depth = stack.pop()
		node_depth[node_id] = parent_depth + 1
		# If we have a test node
		if (children_left[node_id] != children_right[node_id]):
			stack.append((children_left[node_id], parent_depth + 1))
			stack.append((children_right[node_id], parent_depth + 1))
		else:
			is_leaves[node_id] = True

	output+="The binary tree structure has"+ str(n_nodes)+"nodes and has the following tree structure: \n"
	for i in range(n_nodes):
		if is_leaves[i]:
			output+=(node_depth[i] * "\t")+"node="+str(i)+" leaf node.\n"
		else:
			output+=(node_depth[i] * "\t")+"node="+str(i)+" test node: go to node"+str(children_left[i])+" if "+headers[feature[i]]+" <= "+ str(threshold[i])+" else to node"+str(children_right[i])+"\n"
	#print()

	fileName=projectName+"/TreeStructureDescription/Tree#"+str(treeNumber)+".txt"
	with open(fileName,'w') as f:
		f.write(output)
	f.close()

def printPathAttributes(estimator,X_test,sample_id,headers):
	node_indicator = estimator.decision_path(X_test)
	leave_id = estimator.apply(X_test)
	node_index = node_indicator.indices[node_indicator.indptr[sample_id]:node_indicator.indptr[sample_id + 1]]
	n_nodes = estimator.tree_.node_count
	children_left = estimator.tree_.children_left
	children_right = estimator.tree_.children_right
	feature = estimator.tree_.feature
	threshold = estimator.tree_.threshold

	print('Rules used to predict sample %s: ' % sample_id)
	for node_id in node_index:
		if leave_id[sample_id] != node_id:
			continue
		if (float(X_test[sample_id][feature[node_id]]) <= threshold[node_id]):
			threshold_sign = "<="
		else:
			threshold_sign = ">"
		print("decision id node %s : (X_test[%s, %s] (= %s) %s %s)"% (node_id,sample_id,feature[node_id],X_test[sample_id][feature[node_id]],threshold_sign,threshold[node_id]))
	print("Done.")
def returnPathAttributes(estimator,X_test,sample_id):
	node_indicator = estimator.decision_path(X_test)
	leave_id = estimator.apply(X_test)
	attributesUsedInDecisionPath=[]
	node_index = node_indicator.indices[node_indicator.indptr[sample_id]:node_indicator.indptr[sample_id + 1]]
	for node_id in node_index:
		if leave_id[sample_id] != node_id:
			continue
		attributesUsedInDecisionPath.append(node_id)
	return attributesUsedInDecisionPath

def loadModel(filename):
	model=joblib.load(filename)
	return model

def loadData(testXName, testYName, trainXName, trainYName,testXHeader,testYHeader,trainXHeader,trainYHeader):
	if(testXHeader==1):
		testX=pd.read_csv(testXName)
		headers=list(testX.columns.values)
	else:
		testX=pd.read_csv(testXName,header=None)

	if(testYHeader==1):
		testY=pd.read_csv(testYName)
	else:
		testY=pd.read_csv(testYName,header=None)

	if(trainXHeader==1):
		trainX=pd.read_csv(trainXName)
		headers=list(trainX.columns.values)
	else:
		trainX=pd.read_csv(trainXName,header=None)

	if(trainYHeader==1):
		trainY=pd.read_csv(trainYName)
	else:
		trainY=pd.read_csv(trainYName,header=None)
	
	return testX,testY,trainX,trainY,headers

def returnValueFunc(a):
	return a[1]

def createSubDirectory(mainDir, subDir):
	os.mkdir(mainDir+"/"+subDir)

def makeGraph(filename,noToDisplay,graphTitle,projectName):
	fp=open(filename)
	data=json.load(fp)

	keys=list(data.keys())
	values=list(data.values())

	if(len(keys)<noToDisplay):
		pass
	else:
		keys=[]
		values=[]
		dataNew=[]
		i=0
		for key in data:
			dataNew.append([key,data[key]])
		#print(sorted(dataNew, key=returnValueFunc),reverse=True)
		for key, value in sorted(dataNew, key=returnValueFunc,reverse=True):
			if(i<noToDisplay):
				keys.append(key)
				values.append(value)
			else:
				pass
			i+=1
	yPos=np.arange(len(keys))
	plt.figure(figsize=(15, 20))
	plt.barh(yPos, values, alpha=1)
	plt.yticks(yPos, keys)
	plt.xlabel('Average Feature Contribution Per Tree Per Sample')
	plt.title(graphTitle)
	imageFileName=projectName+"/Graphs/"+str(graphTitle)+".jpeg"	
	plt.savefig(imageFileName)

def createGraphs(titles,fileNamesJSON,numberOfAttributesInGraph,projectName):
	createSubDirectory(projectName,"Graphs")
	for i in range(4):
		makeGraph(fileNamesJSON[i],numberOfAttributesInGraph,titles[i],projectName)

def makeGraphsForTestData(testX,testY,model,headers,projectName,numberOfAttributesInGraph):
	predictions=model.predict(testX)
	falsePositive={}
	falseNegative={}
	truePositive={}
	trueNegative={}

	testY=np.array(testY)

	print()

	print("Starting predictions on test data.")

	toolbarWidth=40
	print("Progress ->",end=" ")
	sys.stdout.write("[%s]" % (" " * toolbarWidth))
	sys.stdout.flush()
	sys.stdout.write("\b" * (toolbarWidth+1))
	
	#print(testY[0])
	l=len(predictions)
	mult=1
	prod=l//toolbarWidth
	for i in range(len(predictions)):
		predicted=predictions[i]
		actual=float(testY[i][0])
		if(actual==predicted):
			if(actual==1):
				for j in range(len(model.estimators_)):
					attributesInvolved=returnPathAttributes(model.estimators_[j],testX,i)
					for k in attributesInvolved:
						if(headers[k] not in truePositive):
							truePositive[headers[k]]=1
						else:
							truePositive[headers[k]]+=1
			else:
				for j in range(len(model.estimators_)):
					attributesInvolved=returnPathAttributes(model.estimators_[j],testX,i)
					for k in attributesInvolved:
						if(headers[k] not in trueNegative):
							trueNegative[headers[k]]=1
						else:
							trueNegative[headers[k]]+=1
		else:
			if(actual==1):
				for j in range(len(model.estimators_)):
					attributesInvolved=returnPathAttributes(model.estimators_[j],testX,i)
					for k in attributesInvolved:
						if(headers[k] not in falseNegative):
							falseNegative[headers[k]]=1
						else:
							falseNegative[headers[k]]+=1
			else:
				for j in range(len(model.estimators_)):
					attributesInvolved=returnPathAttributes(model.estimators_[j],testX,i)
					for k in attributesInvolved:
						if(headers[k] not in falsePositive):
							falsePositive[headers[k]]=1
						else:
							falsePositive[headers[k]]+=1
		#print("Done for ",i,"/",len(predictions))

		if(i==prod*mult):
			sys.stdout.write("|")
			sys.stdout.flush()
			mult+=1

	for key in falsePositive:
		#no of times it occurs per tree in the model
		falsePositive[key]=falsePositive[key]/len(model.estimators_)
		#divide by number of test samples
		falsePositive[key]=falsePositive[key]/len(testX)
	for key in falseNegative:
		#no of times it occurs per tree in the model
		falseNegative[key]=falseNegative[key]/len(model.estimators_)
		#divide by number of test samples
		falseNegative[key]=falseNegative[key]/len(testX)
	for key in truePositive:
		#no of times it occurs per tree in the model
		truePositive[key]=truePositive[key]/len(model.estimators_)
		#divide by number of test samples
		truePositive[key]=truePositive[key]/len(testX)
	for key in trueNegative:
		#no of times it occurs per tree in the model
		trueNegative[key]=trueNegative[key]/len(model.estimators_)
		#divide by number of test samples
		trueNegative[key]=trueNegative[key]/len(testX)

	falsePositiveSum=sum(falsePositive.values())
	for key in falsePositive:
		falsePositive[key]=falsePositive[key]/falsePositiveSum
		falsePositive[key]="{:.2f}".format(falsePositive[key])
		falsePositive[key]=float(falsePositive[key])

	falseNegativeSum=sum(falseNegative.values())
	for key in falseNegative:
		falseNegative[key]=falseNegative[key]/falseNegativeSum
		falseNegative[key]="{:.2f}".format(falseNegative[key])
		falseNegative[key]=float(falseNegative[key])

	truePositiveSum=sum(truePositive.values())
	for key in truePositive:
		truePositive[key]=truePositive[key]/truePositiveSum
		truePositive[key]="{:.2f}".format(truePositive[key])
		truePositive[key]=float(truePositive[key])

	trueNegativeSum=sum(trueNegative.values())
	for key in trueNegative:
		trueNegative[key]=trueNegative[key]/trueNegativeSum
		trueNegative[key]="{:.2f}".format(trueNegative[key])
		trueNegative[key]=float(trueNegative[key])

	createSubDirectory(projectName,"JSONFiles")
	falsePositiveJSONFileName=projectName+"/JSONFiles/falsePositive.json"
	with open(falsePositiveJSONFileName, 'w') as fp1:
		json.dump(falsePositive, fp1,indent=1)
	fp1.close()

	falseNegativeJSONFileName=projectName+"/JSONFiles/falseNegative.json"
	with open(falseNegativeJSONFileName, 'w') as fp1:
		json.dump(falseNegative, fp1,indent=1)
	fp1.close()

	truePositiveJSONFileName=projectName+"/JSONFiles/truePositive.json"
	with open(truePositiveJSONFileName, 'w') as fp1:
		json.dump(truePositive, fp1,indent=1)
	fp1.close()

	trueNegativeJSONFileName=projectName+"/JSONFiles/trueNegative.json"
	with open(trueNegativeJSONFileName, 'w') as fp1:
		json.dump(trueNegative, fp1,indent=1)
	fp1.close()

	titles=["False Positive","False Negative","True Positive","True Negative"]
	fileNamesJSON=[falsePositiveJSONFileName,falseNegativeJSONFileName,truePositiveJSONFileName,trueNegativeJSONFileName]

	createGraphs(titles,fileNamesJSON,numberOfAttributesInGraph,projectName)

def printDetails(sleepTime):
	print("Libraries Needed Apart From the Normal Ones : 1.graphviz\n")
	print("All the pathnames that are asked for should be entered relative to the current working directory.\n")
	time.sleep(sleepTime)
	print("The data must be split into 4 files which are : ")
	time.sleep(sleepTime)
	print("1.Test Data Attributes Values which are X Values for Test Data")
	time.sleep(sleepTime)
	print("2.Test Data Output which are Y Values for Test Data")
	time.sleep(sleepTime)
	print("3.Train Data Attributes Values which are X Values for Train Data")
	time.sleep(sleepTime)
	print("4.Train Data Output which are Y Values for Train Data")
	time.sleep(sleepTime)
	print()
	print("All the data must be in the CSV Format and atleast one of the two : Test Data X or Train Data X must have the headers which are the corresponding Attribute Names.")
	print()
	time.sleep(sleepTime)
	print("Model should be stored in a pickle file.")

def getNames():
	print("Lets start taking the names of the data files and the model.")
	projectName=input("Give me a name for your project : ")
	testX=input("Enter the name of the Test X CSV Data File : ")
	testXHeader=int(input("Enter 1 if Test X has Headers else enter 0 : "))
	print()
	trainX=input("Enter the name of the Train X CSV Data File : ")
	trainXHeader=int(input("Enter 1 if Train X has Headers else enter 0: "))
	print()
	testY=input("Enter the name of the Test Y CSV Data File : ")
	testYHeader=int(input("Enter 1 if Test Y has Headers else enter 0 :"))
	print()
	trainY=input("Enter the name of the Train Y CSV Data File : ")
	trainYHeader=int(input("Enter 1 if Train Y has Headers else enter 0 :"))

	print()

	if(testXHeader==0 and trainXHeader==0):
		print("Atleast one of the two TestX Csv or TrainX Csv must have the headers in them. Please change them and re-enter")
		time.sleep(2)
		projectName,testX,testY,trainX,trainY,testXHeader,testYHeader,trainXHeader,trainYHeader=getNames()
		return projectName,testX,testY,trainX,trainY,testXHeader,testYHeader,trainXHeader,trainYHeader

	numberOfAttributesInGraph=int(input("How many attributes are required in the graphs? : "))

	return projectName,testX,testY,trainX,trainY,testXHeader,testYHeader,trainXHeader,trainYHeader,numberOfAttributesInGraph

def loadModelWithFilename():
	fileNameForModel=input("Enter the File Name for the Model to be loaded: ")
	model=loadModel(fileNameForModel)
	return model

def createDirectoryForProject(projectName):
	if(os.path.isdir("./"+projectName)):
		shutil.rmtree(projectName)
	os.mkdir(projectName)

def generateTreesPDF(model,projectName):
	i=1
	for t in model.estimators_:
		dot_data = tree.export_graphviz(t, out_file=None) 
		graph = graphviz.Source(dot_data)
		fname=projectName+"/Trees/Tree#"+str(i)+".pdf" 
		graph.render(fname)
		i+=1
	path=projectName+"/Trees/"
	l=os.listdir(path)
	for file in l:
		if(len(file.split('.'))==2):
			os.remove(path+file)

def generateAttributeContribution(model,projectName,headers):
	featureImp=pd.Series(model.feature_importances_,index=headers).sort_values(ascending=False)
	op=featureImp.to_string()

	allFeaturesFileName=projectName+"/AttributesContribution/allFeatures.txt"
	with open(allFeaturesFileName,'w') as f:
		f.write(op)
	f.close()

	lines=op.split("\n")
	newOp=""
	notDone=True
	i=0
	l=len(lines)
	while(notDone==True and i<l):
		lineString=lines[i]
		line=lineString.strip().split()
		#print(line)	
		if(float(line[1])==0):
			notDone=False
		else:
			newOp+=lineString+"\n"
		i+=1
	
	onlyNonNegativeFeaturesFileName=projectName+"/AttributesContribution/featuresWithNonNegativeContribution.txt"
	with open(onlyNonNegativeFeaturesFileName,'w') as f:
		f.write(newOp)
	f.close()

def generateTreesDescription(model,projectName,headers):
	i=1
	for tree in model.estimators_:
		printTreeStructure(tree,headers,i,projectName)
		i+=1

def makeREADME(projectName,numberOfAttributesInGraph,numOfTrees):
	filename=projectName+"/README.txt"
	op="Understanding the Output.\n\n"
	op+="The output will be divided into 6 folders under the "+projectName+" directory.\n"
	op+="They are : \n"
	
	op+="1.Attributes Contribution :\n\t->This files in this folder are intended to explain the attribute contributions averaged over the entire Random Forest."
	op+="\n\t->From among the two files, allFeatures.txt gives the features and their contribution for all the features."
	op+="\n\t->featuresWithNonNegativeContribution.txt gives the features with only Non Zero Contributions."

	op+="\n2.Graphs : \n\t-> The four graphs here denote the top "+str(numberOfAttributesInGraph)+" features present in the decision paths of all the test samples for each of the 4 classes."

	op+="\n3.JSONFiles : \n\t-> The four JSON Files here store the count of all the attributes appearing in the decision paths of all the test samples for each of the four classes."

	op+="\n4.Trees : \n\t-> The files in this folder are the pictorial representations of the "+str(numOfTrees)+" trees in the Random Forest Classifier."

	op+="\n5.Tree Structure Description : \n\t-> The files in this folder are the text representations of the "+str(numOfTrees)+" trees in the Random Forest Classifier depicting \n\teach of the decision levels in the tree."

	op+="\n6. Paths: \n\t-> The files in this folder are the decision paths for each sample for each tree in the model."

	with open(filename,'w') as f:
		f.write(op)
	f.close()

def main():
	sleepTime=2
	print("Hello.\nBefore we start, here are a few details you might want to know : \n")
	time.sleep(sleepTime)
	
	printDetails(2)
	
	projectName,testXName,testYName,trainXName,trainYName,testXHeader,testYHeader,trainXHeader,trainYHeader,numberOfAttributesInGraph=getNames()

	createDirectoryForProject(projectName)
	testX,testY,trainX,trainY,headers=loadData(testXName,testYName,trainXName,trainYName,testXHeader,testYHeader,trainXHeader,trainYHeader)
	
	model=loadModelWithFilename()

	createSubDirectory(projectName,"TreeStructureDescription")
	generateTreesDescription(model,projectName,headers)

	createSubDirectory(projectName,"Trees")
	generateTreesPDF(model,projectName)

	createSubDirectory(projectName,"AttributesContribution")
	generateAttributeContribution(model,projectName,headers)

	printPathsIntoFile(testX,model,projectName,headers)
	
	makeGraphsForTestData(testX,testY,model,headers,projectName,numberOfAttributesInGraph)
	makeREADME(projectName,numberOfAttributesInGraph,len(model.estimators_))
main()
