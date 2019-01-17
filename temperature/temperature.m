clc
clear all

%==========================================================================
%                         Data import using csvimport                     =
%==========================================================================
%Load all data
[date meanTemperature] = csvimport( 'mean-daily-temperature-fisher-ri.csv', 'columns', [1,2],'noHeader', true); 
%convert rows to collumns
date = date';
meanTemperature = meanTemperature';
%delete first collumns
date(:,1)=[];
meanTemperature(:,1)=[];

%==========================================================================
%        Creation of Training 70%, validation 15% and test 15%            =
%==========================================================================

trainDate = date(:,[1:1024]);
validationDate = date(:,[1025:1243]);
testDate = date(:,[1244:1462]);

trainTemperature = meanTemperature(:,[1:1024]);
validationTemperature = meanTemperature(:,[1025:1243]);
testTemperature = meanTemperature(:,[1244:1462]);

