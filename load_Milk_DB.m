function [Month,MonthlymilkproductionpoundspercowJan62Dec75adjustedformonthleng] = importfile(filename, startRow, endRow)
%IMPORTFILE Import numeric data from a text file as column vectors.
%   [MONTH,MONTHLYMILKPRODUCTIONPOUNDSPERCOWJAN62DEC75ADJUSTEDFORMONTHLENG]
%   = IMPORTFILE(FILENAME) Reads data from text file FILENAME for the
%   default selection.
%
%   [MONTH,MONTHLYMILKPRODUCTIONPOUNDSPERCOWJAN62DEC75ADJUSTEDFORMONTHLENG]
%   = IMPORTFILE(FILENAME, STARTROW, ENDROW) Reads data from rows STARTROW
%   through ENDROW of text file FILENAME.
%
% Example:
%   [Month,MonthlymilkproductionpoundspercowJan62Dec75adjustedformonthleng] = importfile('monthly-milk-production-pounds-p.csv',2, 157);
%
%    See also TEXTSCAN.

% Auto-generated by MATLAB on 2019/01/21 19:57:20

%% Initialize variables.
delimiter = ',';
if nargin<=2
    startRow = 2;
    endRow = 157;
end

%% Format for each line of text:
%   column1: datetimes (%{yyyy-MM}D)
%	column2: double (%f)
% For more information, see the TEXTSCAN documentation.
formatSpec = '%{yyyy-MM}D%f%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to the format.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
textscan(fileID, '%[^\n\r]', startRow(1)-1, 'WhiteSpace', '', 'ReturnOnError', false);
dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'TextType', 'string', 'ReturnOnError', false, 'EndOfLine', '\r\n');
for block=2:length(startRow)
    frewind(fileID);
    textscan(fileID, '%[^\n\r]', startRow(block)-1, 'WhiteSpace', '', 'ReturnOnError', false);
    dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'TextType', 'string', 'ReturnOnError', false, 'EndOfLine', '\r\n');
    for col=1:length(dataArray)
        dataArray{col} = [dataArray{col};dataArrayBlock{col}];
    end
end

%% Close the text file.
fclose(fileID);

%% Post processing for unimportable data.
% No unimportable data rules were applied during the import, so no post
% processing code is included. To generate code which works for
% unimportable data, select unimportable cells in a file and regenerate the
% script.

%% Allocate imported array to column variable names
Month = dataArray{:, 1};
MonthlymilkproductionpoundspercowJan62Dec75adjustedformonthleng = dataArray{:, 2};

% For code requiring serial dates (datenum) instead of datetime, uncomment
% the following line(s) below to return the imported dates as datenum(s).

% Month=datenum(Month);


