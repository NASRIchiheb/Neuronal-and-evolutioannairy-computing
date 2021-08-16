%Import dataset------------------------------------
data = importfile('A1-turbine.txt', 3);
%Separate dataset----------------------------------
dataTrain = data(1:401,:);
dataTest =  data(401:451,:);
%Load variables for MLR----------------------------
x = [dataTrain.Powe,dataTrain.rOfA,dataTrain.hydroel,dataTrain.ectric];
y = [dataTrain.alTurbine];
%Create linear regression model to fit with data X
mdl = fitlm(x,y);
%Prepare data for test-----------------------------
xTest = [dataTest.Powe,dataTest.rOfA,dataTest.hydroel,dataTest.ectric];
yTest = [dataTest.alTurbine];
%Predict turbin power from test set----------------
ypred = predict(mdl,xTest);

%Do some plots-------------------------------------

plot(yTest,ypred,'.');
xlabel("Test data");
ylabel("Prediction");
title("MLR: Test data according to predictions");

%Calculate error rate------------------------------
error_rate = 100 * ((sum(abs(ypred-yTest)))/sum(yTest));
disp("Error rate MLR : "+error_rate + "%")



    
