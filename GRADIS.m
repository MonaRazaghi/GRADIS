% GRADIS algorithm on data with available negative samples, so training negative samples is not needed
% Expression file contains the gene expressions of the genes in Genes.xlsx file
% The Network.xlsx file contains three columns. First column is the TF, second column is the target and third column indicates the existence of a regulation

clear, clc

%% Loading data

EXPfilename = 'Expression.txt';
GENEfilename = 'Genes.xlsx';
NETfilename = 'Network.xlsx';

disp('Loading data...     (takes a while)');

T = readtable(EXPfilename);
X = table2array(T);

[~,ExpGenes,~] = xlsread(GENEfilename,'A:A');

[~,NetTF,~] = xlsread(NETfilename, 'A:A');
[~,NetGenes,~] = xlsread(NETfilename, 'B:B');
PosOrNeg = xlsread(NETfilename, 'C:C');

disp('Loading data ...     (Done!)')

%% Initializing
UniNetTF = unique(NetTF);
UniNetGene = union(unique(NetGenes),UniNetTF);

[C1,iNetGenes,iExpGenes] = intersect(UniNetGene,ExpGenes, 'stable');

XX = X(iExpGenes,:);

[GeneNumber,TimeNumber] = size(XX);

% k : number of clusters
k = floor((1 + sqrt(4 * TimeNumber + 1))/2);

[~,XX_Clustered] = kmeans(XX',k);
XX_Clustered = XX_Clustered';

rowmax = max(XX_Clustered(:,:),[],2);
XX_scaled = (XX_Clustered(:,:))./rowmax;

ExpGenes = ExpGenes(iExpGenes);

[C2,iTF_Genes,iTF] = intersect(ExpGenes,UniNetTF, 'stable');
TFsInExp = ExpGenes(iTF_Genes);
inonTF_Genes = setdiff(1:numel(iExpGenes),iTF_Genes);
inonTF_Genes = inonTF_Genes';
nonTFsInExp = ExpGenes(inonTF_Genes);
num_TF = numel(iTF_Genes);
num_nonTF = numel(inonTF_Genes);

%
%% Network initialization

NetPos = zeros(numel(ExpGenes));

for i = 1:numel(NetTF)
    if find(strcmp(ExpGenes,NetTF{i}))
        if find(strcmp(ExpGenes,NetGenes{i}))
            if PosOrNeg(i) == 1
                NetPos(find(strcmp(ExpGenes,NetTF{i})),find(strcmp(ExpGenes,NetGenes{i}))) = 1; % Weights(i);
            end
        end
    end
end

NetworkPos = NetPos(iTF_Genes,:);


NetNeg = zeros(numel(ExpGenes));

for i = 1:numel(NetTF)
    if find(strcmp(ExpGenes,NetTF{i}))
        if find(strcmp(ExpGenes,NetGenes{i}))
            if PosOrNeg(i) == 0
                NetNeg(find(strcmp(ExpGenes,NetTF{i})),find(strcmp(ExpGenes,NetGenes{i}))) = 1; % Weights(i);
            end
        end
    end
end

NetworkNeg = NetNeg(iTF_Genes,:);

[r_Neg,c_Neg] = find(NetworkNeg == 1);
[r_Pos,c_Pos] = find(NetworkPos == 1);

Num_Features = ((size(XX_scaled,2)-1)*(size(XX_scaled,2)))/2;


%% Positives

disp('Feature construction...     (takes a while)');

Data_Pos = zeros(length(r_Pos),Num_Features);

for r = 1:length(r_Pos)
    
    X_pos = XX_scaled(iTF_Genes(r_Pos(r)),1:end);
    Y_pos = XX_scaled(c_Pos(r),1:end);
    
    Dist_pos = zeros(length(X_pos));
    
    for i = 1:length(X_pos)
        for j = i+1: length(X_pos)
            Dist_pos(i,j) = ((X_pos(i)-X_pos(j))^2 + (Y_pos(i)-Y_pos(j))^2)^(0.5);
            Dist_pos(j,i) = ((X_pos(i)-X_pos(j))^2 + (Y_pos(i)-Y_pos(j))^2)^(0.5);
        end
    end
    
    inx = 1;
    for i = 1:size(Dist_pos,2)
        Data_Pos(r,inx:inx+size(Dist_pos,2)-i-1) = Dist_pos(i,i+1:end);
        inx = inx+size(Dist_pos,2)-i;
    end
    
end

%% Negatives

rand_num_Neg = randperm(length(r_Neg));
Data_Neg = zeros(length(r_Pos),Num_Features); %% for memory

for r = 1:length(r_Pos) %% for memory
    
    X_neg = XX_scaled(iTF_Genes(r_Neg(rand_num_Neg(r))),1:end);
    Y_neg = XX_scaled(c_Neg(rand_num_Neg(r)),1:end);
    
    Dist_neg = zeros(length(X_neg));
    
    for i = 1:length(X_neg)
        for j = i+1: length(X_neg)
            Dist_neg(i,j) = ((X_neg(i)-X_neg(j))^2 + (Y_neg(i)-Y_neg(j))^2)^(0.5);
            Dist_neg(j,i) = ((X_neg(i)-X_neg(j))^2 + (Y_neg(i)-Y_neg(j))^2)^(0.5);
        end
    end
    
    inx = 1;
    for i = 1:size(Dist_neg,2)
        Data_Neg(r,inx:inx+size(Dist_neg,2)-i-1) = Dist_neg(i,i+1:end);
        inx = inx+size(Dist_neg,2)-i;
    end
    
end

disp('Feature construction...     (Done)');

%% Final svm

Train_number = floor(9 * length(r_Pos)/10);

rand_num_Pos = randperm(size(Data_Pos,1));
rand_Neg = randperm(size(Data_Neg,1));

Data_Train = [Data_Pos(rand_num_Pos(1:Train_number),:) ; Data_Neg(rand_Neg(1:Train_number),:)];

Group_Train = [ones(Train_number,1) ; zeros(Train_number,1)];

SVMModel = fitcsvm(Data_Train,Group_Train,'Holdout',0.1,'Standardize',true,'KernelFunction','rbf',...
    'KernelScale','auto');
CompactSVMModel = SVMModel.Trained{1}; % Extract trained, compact classifier


% Test
Data_Test = [Data_Pos(rand_num_Pos(Train_number+1:end),:) ; Data_Neg(rand_Neg(Train_number+1:length(r_Pos)),:)];
Test_number = length(r_Pos) - Train_number;
Group_Test = [ones(Test_number,1) ; zeros(Test_number,1)];

accuracy = sum(predict(CompactSVMModel, Data_Test) == Group_Test)/length(Group_Test)*100;

[label,score] = predict(CompactSVMModel,Data_Test);

[X_AUC,Y_AUC,Tsvm,AUCsvm] = perfcurve(logical(Group_Test),score(:,logical(CompactSVMModel.ClassNames)),'true');
[X_PR,Y_PR,~,PRsvm] = perfcurve(logical(Group_Test),score(:,logical(CompactSVMModel.ClassNames)),'true','xCrit', 'reca', 'yCrit', 'prec');

%% Display results

STR1 = ['The area under ROC is : ', num2str(AUCsvm)];
disp(STR1);    

STR2 = ['The area under PR curve is : ', num2str(PRsvm)];
disp(STR2);

figure(1)
hold on
t = linspace(realmin ( 'single' ),1);
plot(t,t,'--','Color',[0,0.45,0.74]);
plot(X_AUC,Y_AUC,'Color',[0.5,0.5,0.5]);

xlabel('False positive rate')
ylabel('True positive rate')
title('ROC')

figure(2)
hold on
t = linspace(realmin ( 'single' ),1);
plot(t,1-t,'--','Color',[0,0.45,0.74]);
plot(X_PR,Y_PR,'Color',[0.5,0.5,0.5]);

xlabel('Recall')
ylabel('Precision')
title('PR')
