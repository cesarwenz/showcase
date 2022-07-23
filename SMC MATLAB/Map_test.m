%====================================================================
% Author: Cesar Wen Zhu
% Date: April 13, 2021
%====================================================================
% Testing file
clear all; close all;

% Test 1
ox = [0.0, 20.0, 50.0, 100.0, 130.0, 40.0, 0.0];
oy = [0.0, -20.0, 0.0, 30.0, 60.0, 80.0, 0.0];
resolution = 2;
row_width = 10;
offset = 1;
radius = 16;
path = RowPlanning(ox, oy, resolution,...
                row_width, offset, radius);
path.plot_path();

% Test 2
% ox = [0.0, 20.0, 50.0, 100.0, 130.0, 40.0, 0.0];
% oy = [0.0, -20.0, 0.0, 30.0, 60.0, 80.0, 0.0];
% resolution = 5.0;
% row_width = 10;
% offset = 2;
% radius = 10;
% path = RowPlanning(ox, oy, resolution,...
%                 row_width, offset, radius);
% path.plot_path();

% Test 3
% ox= [0.0, 10.0, 10.0, 0.0, 0.0];
% oy = [0.0, 0.0, 10.0, 10.0, 0.0];
% resolution = 0.5;
% row_width = 3;
% offset = 3;
% radius = 5;
% path = RowPlanning(ox, oy, resolution,...
%                 row_width, offset, radius);
% path.plot_path();