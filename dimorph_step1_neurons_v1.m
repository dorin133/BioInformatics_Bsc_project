tic
clear all
close all
addpath('/data/matlab_functions/')

savefig_flag = 0;
savefig_pdf = 0;
lth = 0; hth = 50e3;
mtx_file = '/data/runs/samples/10X35_1/out10X35_1_210219/outs/filtered_feature_bc_matrix/matrix.mtx';
bc_file = '/data/runs/samples/10X35_1/out10X35_1_210219/outs/filtered_feature_bc_matrix/barcodes.tsv';
gene_file = '/data/runs/samples/10X35_1/out10X35_1_210219/outs/filtered_feature_bc_matrix/features.tsv';
[data_35_1, geneid_35_1, barcodes_35_1] = load10xMtxFile(mtx_file,bc_file,gene_file,lth,hth);

mtx_file = '/data/runs/samples/10X35_2/out10X35_2_210219/outs/filtered_feature_bc_matrix/matrix.mtx';
bc_file = '/data/runs/samples/10X35_2/out10X35_2_210219/outs/filtered_feature_bc_matrix/barcodes.tsv';
gene_file = '/data/runs/samples/10X35_2/out10X35_2_210219/outs/filtered_feature_bc_matrix/features.tsv';
[data_35_2, geneid_35_2, barcodes_35_2] = load10xMtxFile(mtx_file,bc_file,gene_file,lth,hth);

mtx_file = '/data/runs/samples/10X36_1/out10X36_1_210219/outs/filtered_feature_bc_matrix/matrix.mtx';
bc_file = '/data/runs/samples/10X36_1/out10X36_1_210219/outs/filtered_feature_bc_matrix/barcodes.tsv';
gene_file = '/data/runs/samples/10X36_1/out10X36_1_210219/outs/filtered_feature_bc_matrix/features.tsv';
[data_36_1, geneid_36_1, barcodes_36_1] = load10xMtxFile(mtx_file,bc_file,gene_file,lth,hth);

mtx_file = '/data/runs/samples/10X36_2/out10X36_2_210220/outs/filtered_feature_bc_matrix/matrix.mtx';
bc_file = '/data/runs/samples/10X36_2/out10X36_2_210220/outs/filtered_feature_bc_matrix/barcodes.tsv';
gene_file = '/data/runs/samples/10X36_2/out10X36_2_210220/outs/filtered_feature_bc_matrix/features.tsv';
[data_36_2, geneid_36_2, barcodes_36_2] = load10xMtxFile(mtx_file,bc_file,gene_file,lth,hth);

mtx_file = '/data/runs/samples/10X37_1/out10X37_1_210220/outs/filtered_feature_bc_matrix/matrix.mtx';
bc_file = '/data/runs/samples/10X37_1/out10X37_1_210220/outs/filtered_feature_bc_matrix/barcodes.tsv';
gene_file = '/data/runs/samples/10X37_1/out10X37_1_210220/outs/filtered_feature_bc_matrix/features.tsv';
[data_37_1, geneid_37_1, barcodes_37_1] = load10xMtxFile(mtx_file,bc_file,gene_file,lth,hth);

mtx_file = '/data/runs/samples/10X37_2/out10X37_2_210220/outs/filtered_feature_bc_matrix/matrix.mtx';
bc_file = '/data/runs/samples/10X37_2/out10X37_2_210220/outs/filtered_feature_bc_matrix/barcodes.tsv';
gene_file = '/data/runs/samples/10X37_2/out10X37_2_210220/outs/filtered_feature_bc_matrix/features.tsv';
[data_37_2, geneid_37_2, barcodes_37_2] = load10xMtxFile(mtx_file,bc_file,gene_file,lth,hth);

mtx_file = '/data/runs/samples/10X38_1/out10X38_1_210220/outs/filtered_feature_bc_matrix/matrix.mtx';
bc_file = '/data/runs/samples/10X38_1/out10X38_1_210220/outs/filtered_feature_bc_matrix/barcodes.tsv';
gene_file = '/data/runs/samples/10X38_1/out10X38_1_210220/outs/filtered_feature_bc_matrix/features.tsv';
[data_38_1, geneid_38_1, barcodes_38_1] = load10xMtxFile(mtx_file,bc_file,gene_file,lth,hth);

mtx_file = '/data/runs/samples/10X38_2/out10X38_2_210220/outs/filtered_feature_bc_matrix/matrix.mtx';
bc_file = '/data/runs/samples/10X38_2/out10X38_2_210220/outs/filtered_feature_bc_matrix/barcodes.tsv';
gene_file = '/data/runs/samples/10X38_2/out10X38_2_210220/outs/filtered_feature_bc_matrix/features.tsv';
[data_38_2, geneid_38_2, barcodes_38_2] = load10xMtxFile(mtx_file,bc_file,gene_file,lth,hth);

mtx_file = '/data/runs/samples/10X51_1/out10X51_1_210220/outs/filtered_feature_bc_matrix/matrix.mtx';
bc_file = '/data/runs/samples/10X51_1/out10X51_1_210220/outs/filtered_feature_bc_matrix/barcodes.tsv';
gene_file = '/data/runs/samples/10X51_1/out10X51_1_210220/outs/filtered_feature_bc_matrix/features.tsv';
[data_51_1, geneid_51_1, barcodes_51_1] = load10xMtxFile(mtx_file,bc_file,gene_file,lth,hth);

mtx_file = '/data/runs/samples/10X51_2/out10X51_2_210220/outs/filtered_feature_bc_matrix/matrix.mtx';
bc_file = '/data/runs/samples/10X51_2/out10X51_2_210220/outs/filtered_feature_bc_matrix/barcodes.tsv';
gene_file = '/data/runs/samples/10X51_2/out10X51_2_210220/outs/filtered_feature_bc_matrix/features.tsv';
[data_51_2, geneid_51_2, barcodes_51_2] = load10xMtxFile(mtx_file,bc_file,gene_file,lth,hth);

mtx_file = '/data/runs/samples/10X51_3/out10X51_3_210220/outs/filtered_feature_bc_matrix/matrix.mtx';
bc_file = '/data/runs/samples/10X51_3/out10X51_3_210220/outs/filtered_feature_bc_matrix/barcodes.tsv';
gene_file = '/data/runs/samples/10X51_3/out10X51_3_210220/outs/filtered_feature_bc_matrix/features.tsv';
[data_51_3, geneid_51_3, barcodes_51_3] = load10xMtxFile(mtx_file,bc_file,gene_file,lth,hth);

mtx_file = '/data/runs/samples/10X51_4/out10X51_4_200707/outs/filtered_feature_bc_matrix/matrix.mtx';
bc_file = '/data/runs/samples/10X51_4/out10X51_4_200707/outs/filtered_feature_bc_matrix/barcodes.tsv';
gene_file = '/data/runs/samples/10X51_4/out10X51_4_200707/outs/filtered_feature_bc_matrix/features.tsv';
[data_51_4, geneid_51_4, barcodes_51_4] = load10xMtxFile(mtx_file,bc_file,gene_file,lth,hth);

mtx_file = '/data/runs/samples/10X52_1/out10X52_1_200707/outs/filtered_feature_bc_matrix/matrix.mtx';
bc_file = '/data/runs/samples/10X52_1/out10X52_1_200707/outs/filtered_feature_bc_matrix/barcodes.tsv';
gene_file = '/data/runs/samples/10X52_1/out10X52_1_200707/outs/filtered_feature_bc_matrix/features.tsv';
[data_52_1, geneid_52_1, barcodes_52_1] = load10xMtxFile(mtx_file,bc_file,gene_file,lth,hth);

mtx_file = '/data/runs/samples/10X52_2/out10X52_2_200707/outs/filtered_feature_bc_matrix/matrix.mtx';
bc_file = '/data/runs/samples/10X52_2/out10X52_2_200707/outs/filtered_feature_bc_matrix/barcodes.tsv';
gene_file = '/data/runs/samples/10X52_2/out10X52_2_200707/outs/filtered_feature_bc_matrix/features.tsv';
[data_52_2, geneid_52_2, barcodes_52_2] = load10xMtxFile(mtx_file,bc_file,gene_file,lth,hth);

mtx_file = '/data/runs/samples/10X52_3/out10X52_3_200707/outs/filtered_feature_bc_matrix/matrix.mtx';
bc_file = '/data/runs/samples/10X52_3/out10X52_3_200707/outs/filtered_feature_bc_matrix/barcodes.tsv';
gene_file = '/data/runs/samples/10X52_3/out10X52_3_200707/outs/filtered_feature_bc_matrix/features.tsv';
[data_52_3, geneid_52_3, barcodes_52_3] = load10xMtxFile(mtx_file,bc_file,gene_file,lth,hth);

mtx_file = '/data/runs/samples/10X52_4/out10X52_4_210220/outs/filtered_feature_bc_matrix/matrix.mtx';
bc_file = '/data/runs/samples/10X52_4/out10X52_4_210220/outs/filtered_feature_bc_matrix/barcodes.tsv';
gene_file = '/data/runs/samples/10X52_4/out10X52_4_210220/outs/filtered_feature_bc_matrix/features.tsv';
[data_52_4, geneid_52_4, barcodes_52_4] = load10xMtxFile(mtx_file,bc_file,gene_file,lth,hth);

mtx_file = '/data/runs/samples/10X54_1/out10X54_1_200707/outs/filtered_feature_bc_matrix/matrix.mtx';
bc_file = '/data/runs/samples/10X54_1/out10X54_1_200707/outs/filtered_feature_bc_matrix/barcodes.tsv';
gene_file = '/data/runs/samples/10X54_1/out10X54_1_200707/outs/filtered_feature_bc_matrix/features.tsv';
[data_54_1, geneid_54_1, barcodes_54_1] = load10xMtxFile(mtx_file,bc_file,gene_file,lth,hth);

mtx_file = '/data/runs/samples/10X54_2/out10X54_2_210220/outs/filtered_feature_bc_matrix/matrix.mtx';
bc_file = '/data/runs/samples/10X54_2/out10X54_2_210220/outs/filtered_feature_bc_matrix/barcodes.tsv';
gene_file = '/data/runs/samples/10X54_2/out10X54_2_210220/outs/filtered_feature_bc_matrix/features.tsv';
[data_54_2, geneid_54_2, barcodes_54_2] = load10xMtxFile(mtx_file,bc_file,gene_file,lth,hth);


sample = [repmat({'35-1'},length(barcodes_35_1),1); repmat({'35-2'},length(barcodes_35_2),1);....
    repmat({'36-1'},length(barcodes_36_1),1); repmat({'36-2'},length(barcodes_36_2),1); ....
    repmat({'37-1'},length(barcodes_37_1),1); repmat({'37-2'},length(barcodes_37_2),1); ....
    repmat({'38-1'},length(barcodes_38_1),1); repmat({'38-2'},length(barcodes_38_2),1);....
    repmat({'51-1'},length(barcodes_51_1),1); repmat({'51-2'},length(barcodes_51_2),1);...
    repmat({'51-3'},length(barcodes_51_3),1); repmat({'51-4'},length(barcodes_51_4),1);...
    repmat({'52-1'},length(barcodes_52_1),1); repmat({'52-2'},length(barcodes_52_2),1);...
    repmat({'52-3'},length(barcodes_52_3),1); repmat({'52-4'},length(barcodes_52_4),1);...
    repmat({'54-1'},length(barcodes_54_1),1); repmat({'54-2'},length(barcodes_54_2),1);...
    ];

sample_uni = unique(sample);
data = [data_35_1,data_35_2,data_36_1,data_36_2,data_37_1,data_37_2,data_38_1,data_38_2,....
    data_51_1,data_51_2,data_51_3,data_51_4,data_52_1,data_52_2,data_52_3,data_52_4,data_54_1,data_54_2];
geneid = geneid_35_1;
cellid = [barcodes_35_1;barcodes_35_2;barcodes_36_1;barcodes_36_2;barcodes_37_1;....
    barcodes_37_2;barcodes_38_1;barcodes_38_2;....
    barcodes_51_1;barcodes_51_2;barcodes_51_3;barcodes_51_4;...
    barcodes_52_1;barcodes_52_2;barcodes_52_3;barcodes_52_4;...
    barcodes_54_1;barcodes_54_2];

side_flag = zeros(size(sample));
side_flag(strcmpi(sample,'35-1') | strcmpi(sample,'36-1')....
    | strcmpi(sample,'37-1') | strcmpi(sample,'38-1')) = 1;
side_flag(strcmpi(sample,'35-2') | strcmpi(sample,'36-2')....
    | strcmpi(sample,'37-2') | strcmpi(sample,'38-2')) = -1;

male_flag = strcmpi(sample,'36-1') | strcmpi(sample,'36-2') | strcmpi(sample,'37-1') | strcmpi(sample,'37-2') |....
    strcmpi(sample,'51-3') | strcmpi(sample,'51-4') | strcmpi(sample,'52-3') | strcmpi(sample,'52-4');
female_flag = strcmpi(sample,'35-1') | strcmpi(sample,'35-2') | strcmpi(sample,'38-1') | strcmpi(sample,'38-2') | ...
    strcmpi(sample,'51-1') | strcmpi(sample,'51-2') | strcmpi(sample,'52-1') | strcmpi(sample,'52-2') |...
    strcmpi(sample,'54-1') | strcmpi(sample,'54-2');
parent_flag = strcmpi(sample,'51-3') | strcmpi(sample,'51-4') | strcmpi(sample,'52-3') | strcmpi(sample,'52-4') |....
    strcmpi(sample,'51-1') | strcmpi(sample,'51-2') | strcmpi(sample,'52-1') | strcmpi(sample,'52-2');
cntnp_flag = strcmpi(sample,'54-1') | strcmpi(sample,'54-2');

tot_mol = sum(data);
tot_mol(tot_mol>3e4) = 3e4;
tot_genes = sum(data>0);

% [OLmat,OLmat_frac] = Class_dist_mat(data,geneid,1);

stmn2 = data(strcmpi(geneid,'Stmn2'),:);
snap25 = data(strcmpi(geneid,'Snap25'),:);
% exclude_markers = {'C1qc','C1qa','C1qb','Gja1','Cx3cr1','Acta2','Ly6c1','Mfge8','Plp1'....
%     ,'Aqp4','Vtn','Cldn5','Pdgfrb','Flt1','Slc1a3','Pdgfra','Foxj1','Olig1','Olig2','Sox10','Hbb-bs','Hbb-bt','Hba-a2'};
exclude_markers = {'C1qc','C1qa','C1qb','Gja1','Cx3cr1','Acta2','Ly6c1','Mfge8','Plxnb3','Cldn11'....
    ,'Aqp4','Vtn','Cldn5','Pdgfrb','Flt1','Slc25a18','Pdgfra','Foxj1','Olig1','Olig2','Sox10','Hbb-bs','Hbb-bt','Hba-a2','Ttr'};
[~,loc] = ismember(exclude_markers,geneid);
% validcells = (tot_mol>2500 & tot_mol<5e4 & tot_genes>2000 & sum(data(loc,:)>1)==0 & (stmn2>0 | snap25>0));% only neurons
validcells = (tot_mol>3000 & tot_mol<5e4 & tot_genes>2500); % all cells
for i=1:length(sample_uni)
   fprintf(['valid cells in ',sample_uni{i},' = ', num2str(sum(validcells(strcmpi(sample,sample_uni{i})))),'\n']);
end
sum(validcells)
data = data(:,validcells);
cellid = cellid(validcells);
sample = sample(validcells);
side_flag = side_flag(validcells);
male_flag = male_flag(validcells);
female_flag = female_flag(validcells);
parent_flag = parent_flag(validcells);
cntnp_flag = cntnp_flag(validcells);

cellid = cellfun(@(x,y) [x,'_',y], cellid, sample,'UniformOutput',0);

stmn2 = data(strcmpi(geneid,'Stmn2'),:);
snap25 = data(strcmpi(geneid,'Snap25'),:);
for i=1:length(sample_uni)
   fprintf(['snap25 in before ',sample_uni{i},' = ', num2str(sum(snap25(strcmpi(sample,sample_uni{i}))>0)),'\n']);
end


% save(['afterloading_QC3000_Sdimorph_',date],'data','geneid','cellid','sample','side_flag'....
%     ,'male_flag','female_flag','parent_flag','cntnp_flag','-v7.3');
% load('afterloading_neurons_Sdimorph');
% cellid = cellfun(@(x,y) [x,'_',y], cellid, sample,'UniformOutput',0);
tot_mol = sum(data);
tot_mol(tot_mol>3e4) = 3e4;
tot_genes = sum(data>0);

figure('position',[100,100,800,800],'color','w');
subplot(2,2,1);
[f,xi] = ksdensity(tot_mol);
plot(xi,f);
axis tight;
set(gca,'xlim',[0,prctile(tot_mol,98)]);
xlabel('total mol')
subplot(2,2,3);
[f,xi] = ecdf(tot_mol);
plot(xi,f);
axis tight;
set(gca,'xlim',[0,prctile(tot_mol,98)]);
xlabel('total mol')
subplot(2,2,2);
[f,xi] = ksdensity(tot_genes);
plot(xi,f);
axis tight;
set(gca,'xlim',[0,prctile(tot_genes,98)]);
xlabel('total genes')
subplot(2,2,4);
[f,xi] = ecdf(tot_mol);
plot(xi,f);
axis tight;
set(gca,'xlim',[0,prctile(tot_genes,98)]);
xlabel('total genes')

data = round(data./repmat(sum(data),length(data(:,1)),1)*20e3);

% lev1_cellid = loadCellFile('level1_cellid_cluster_Rep3_21-Apr-2019.txt');
% cluster_annot = loadCellFile_csv('cluster_classs_annot_rep3_all_190421.csv');
% cluster_annot(1,:) = [];
% n_clusters = cell2mat(cluster_annot( strcmpi(cluster_annot(:,2),'neurons'), 1));
% tf = ismember(cell2mat(lev1_cellid(:,2)), n_clusters);
% cellid_n = lev1_cellid(tf,1);
%
% tf = ismember(cellid,cellid_n);
%
% data = data(:,tf);
% cellid = cellid(tf);
% sample = sample(tf);
% side_flag = side_flag(tf);
% male_flag = male_flag(tf);
% female_flag = female_flag(tf);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % %
top_g = 50;
gr1 = find(female_flag);
gr2 = find(male_flag);
x1 = mean(log2(data(:,gr1)+1),2);
x2 = mean(log2(data(:,gr2)+1),2);
d = x1-x2 ;
[~,xi] = sort(d);
figure('position',[200,200,1000,580],'color','w');
[ha, pos] = tight_subplot(1, 2, [0.05,0.05], [0.1,0.05], [0.05,0.05]);
axes(ha(1))
plot(x1, x2, '.');hold on;
xmax = max(x1);
plot([0,xmax],[0,xmax],'-k'); grid on
plot([0,xmax],[0,xmax]+1,'--k'); grid on
plot([1,xmax],[0,xmax-1],'--k'); grid on
plot(x1(xi(1:top_g)), x2(xi(1:top_g)),'.r'); hold on;
text(x1(xi(1:top_g)), x2(xi(1:top_g)),geneid(xi(1:top_g)),'fontsize',6);
xi = flipud(xi);
plot(x1(xi(1:top_g)), x2(xi(1:top_g)),'.r'); hold on;
text(x1(xi(1:top_g)), x2(xi(1:top_g)),geneid(xi(1:top_g)),'fontsize',6);
title(['Sex comparison (average)'])
xlabel(['mean (Females)'])
ylabel(['mean (Males)'])
axis tight

top_g = 50;
x1 = mean(data(:,gr1)>0,2);
x2 = mean(data(:,gr2)>0,2);
d = x1-x2 ;
[~,xi] = sort(d);
axes(ha(2))
plot(x1, x2, '.');hold on;
xmax = max(x1);
plot([0,xmax],[0,xmax],'-k'); grid on
plot([0,xmax],[0,xmax]+0.4,'--k'); grid on
plot([0.4,xmax],[0,xmax-0.4],'--k'); grid on
plot(x1(xi(1:top_g)), x2(xi(1:top_g)),'.r'); hold on;
text(x1(xi(1:top_g)), x2(xi(1:top_g)),geneid(xi(1:top_g)),'fontsize',6);
xi = flipud(xi);
plot(x1(xi(1:top_g)), x2(xi(1:top_g)),'.r'); hold on;
text(x1(xi(1:top_g)), x2(xi(1:top_g)),geneid(xi(1:top_g)),'fontsize',6);
title(['Sex comparison (precentage)'])
xlabel(['mean>0 (Females)'])
ylabel(['mean>0 (Males)'])
axis tight
if savefig_flag==1
    savefig(gcf,['scatter_cluster_',num2str(c1),'_vs_',num2str(c2),'_',date,'.fig'])
end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

in = find(sum(data>0,2)>5 & sum(data>0,2)<length(data(1,:))*0.5 & ~ismember(geneid(:,1),{'Xist','Tsix', 'Eif2s3y', 'Ddx3y', 'Uty', 'Kdm5d'}'));
corr_filt = cv_vs_m_selection(data(in,:),geneid(in),[],1,0);

data_orig_all = data;
geneid_all = geneid;
data = data(in(corr_filt),:);
geneid = geneid(in(corr_filt));

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
moldata = data;


datalog_tmp = cent_norm([(log2(moldata+1))]);
data_tsne = datalog_tmp;
% data_tsne = cent_norm(log2(datamarkers+1));
initial_dims = length(corr_filt);
[prj,m,D,V,Q] = pca_wis(data_tsne',initial_dims);
D = diag(D);
initial_dims = findknee(D);
figure;
subplot(1,2,1)
plot(cumsum(D)); hold on;
plot(initial_dims,sum(D(1:initial_dims)),'sk');
subplot(1,2,2)
plot(D); hold on;
plot(initial_dims,D(initial_dims),'sk');
title(['opt PC = ',num2str(initial_dims)]);
% initial_dims = 30;
prj = prj(:,1:initial_dims);
init = prj(:,1:2)/std(prj(:,1))*1e-4;
init = init-repmat(mean(init),length(init),1);

D = squareform(pdist(prj,'correlation'),'tomatrix');
[Dsort,XI] = sort(D,'ascend');
per_range = 500;
x = 1:per_range;%length(D);
optk = zeros(length(D),1);
for i=1:length(D)
    y = Dsort(1:per_range,i);
    x = x(:);
    y = y(:);
    a = atan((y(end)-y(1))/(x(end)-x(1)));
    xn = x*cos(a) + y*sin(a);
    yn = -x*sin(a) + y*cos(a);
    [~,imax] = max(yn);
    optk(i) = round((x(imax)));
end

perplexity = median(optk);

options = statset('MaxIter',1000);
mapped_xy = tsne(prj,'Algorithm','barneshut','Distance','correlation','NumDimensions',2,'NumPCAComponents',0,.....
    'Perplexity',perplexity,'Standardize',true,'InitialY',init,'LearnRate',500,'Theta',0.5,'Verbose',1,'Options',options,'Exaggeration',20);
toc

% this is just the initial tsne, can be commented later
figure;
set(gcf,'color','w','position',[20,20,900,800])
plot(mapped_xy(:,1),mapped_xy(:,2),'.'); axis tight; axis off
% plot by sample
sample_uni = {'35-1','35-2','36-1','36-2','37-1','37-2','38-1','38-2','51-1','51-2','51-3','51-4','52-2','52-3','52-4','54-1','54-2'};
colors = distinguishable_colors(length(sample_uni)+1);
figure;
set(gcf,'color','w','position',[20,20,1200,1200]);
[ha, pos] = tight_subplot(2, 2, [0.02,0.02], [0.02,0.02], [0.02,0.02]);
axes(ha(1))
for i=1:length(sample_uni)
    s = scatter(mapped_xy(strcmpi(sample,sample_uni{i}),1),mapped_xy(strcmpi(sample,sample_uni{i}),2),10,colors(i,:),'filled'); hold on;
%     alpha(s,0.4);
end
axis tight
axis equal
axis off
legend({'35-1,F,L','35-2,F,R','36-1,M,L','36-2,M,R'...
    ,'37-1,M,L','37-2,M,R','38-1,F,L','38-2,F,R',....
    '51-1,F','51-2,F','51-3,M','51-4,M','52-1,F','52-2,F','52-3,M','52-4,M','54-1,F','54-2,F'})
% plot by left/right
side_flag_uni = [1,-1,0];
colors = [1,0,0;0,0,1,;0,1,0]; %distinguishable_colors(length(sample_uni)+1);
axes(ha(2))
for i=1:length(side_flag_uni)
    s = scatter(mapped_xy(side_flag==side_flag_uni(i),1),mapped_xy(side_flag==side_flag_uni(i),2),10,colors(i,:),'filled'); hold on;
%     alpha(s,0.3);
end
axis tight
axis equal
axis off
legend({'Left','Right','Mixed'})
% plot by male/female
colors = [1,0,0;0,0,1;1,0,0;0,0,1]; %distinguishable_colors(length(sample_uni)+1);
axes(ha(3))
s = scatter(mapped_xy(male_flag,1),mapped_xy(male_flag,2),10,colors(1,:),'filled'); hold on;
% alpha(s,0.3);
s = scatter(mapped_xy(female_flag,1),mapped_xy(female_flag,2),10,colors(2,:),'filled'); hold on;
% alpha(s,0.3);
axis tight
axis off
axis equal
legend({'Male','Female'})
axes(ha(4))
s = scatter(mapped_xy(parent_flag==0,1),mapped_xy(parent_flag==0,2),10,colors(1,:),'filled'); hold on;
% alpha(s,0.3);
s = scatter(mapped_xy(parent_flag,1),mapped_xy(parent_flag,2),10,colors(2,:),'filled'); hold on;
% alpha(s,0.3);
axis tight
axis off
axis equal
legend({'Virgin','Parent'})
title(['PC=',num2str(initial_dims),', perp=',num2str(perplexity),', Ngenes=',num2str(length(corr_filt))])
% if savefig_flag==1
%     savefig(gcf,['tsne_only_neurons_by_annot_v1_perplexity',num2str(perplexity),'_',date,'.fig'])
% %     savefig(gcf,['umap_only_neurons_by_annot_v1_Nneighbors_',num2str(n_neighbors),'_minDist',num2str(min_dist),'_',date,'.fig'])
% end
% % % % % % % % % % % % % % % % % % % % % % % % 
% clustering with dbscan
MinPts = 20;
eps_prc = 70;
[idx, isnoise] = dbscan_epsprc_mipts(mapped_xy,eps_prc,MinPts);
idxuni = unique(idx);

colors = distinguishable_colors(length(unique(idx))+1);
figure;
set(gcf,'color','w','position',[20,20,900,800])
for i=unique(idx)'
    if i>=0
        ii=find(idx==i); h=plot(mapped_xy(ii,1),mapped_xy(ii,2),'o','color',colors(i+1,:),'markersize',3); hold on;
    elseif i>0
        ii=find(idx==i); h=plot(mapped_xy(ii,1),mapped_xy(ii,2),'.','color',colors(i+1,:),'markersize',4); hold on;
    end
end
% for i=idxuni'
%     if i>=0
%         in = idx==i;
%         ht = text(median(mapped_xy(in,1)),median(mapped_xy(in,2)),num2str(i));
%         set(ht,'BackgroundColor',0.8*[1,1,1],'fontsize',6)
%     end
% end
axis tight;
axis equal
axis off

title(['perplexity=',num2str(perplexity),', PC=',num2str(initial_dims),',#C=',num2str(max(idx)),',#out=',num2str(sum(idx==0))],'fontsize',8);

if savefig_flag==1
    savefig(gcf,['tsne_only_neurons_dbscan_v1_perplexity_',num2str(perplexity),date,'.fig'])
end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% sort the data by the clusters and remove outliers
[idx,xi] = sort(idx);
xi(idx==0) = [];
idx(idx==0) = [];
idxuni = unique(idx);
data_sorted_all = data(:,xi);
data_orig_all_sorted = data_orig_all(:,xi);
cellid_sorted = cellid((xi));
sample_sorted = sample((xi));
mapped_xy = mapped_xy(xi,:);
side_flag_sorted = side_flag(xi);
male_flag_sorted = male_flag(xi);
female_flag_sorted = female_flag(xi);
parent_flag_sorted = parent_flag(xi);
cntnp_flag_sorted = cntnp_flag(xi);
prj_sorted = prj(xi,:);

no_dims = 1;
initial_dims = 10;
perplexity = 5;
epsilon = 100;
dist_flag = 2;
theta = 0.5;
rand_seed = 13;
data_tsne = cent_norm(log2(data_sorted_all+1));
xi = [1:length(idx)];
for i=1:length(idxuni)
    i
    ind = find(idx==i);
    if length(ind)>20
        tmp1d = fast_tsne((data_tsne(:,ind))', no_dims, initial_dims, perplexity,theta, rand_seed);
        [~,xitmp] = sort(tmp1d);
        xi(ind) = xi(ind((xitmp)));
    end
end

data_sorted_all = data_sorted_all(:,xi);
data_orig_all_sorted = data_orig_all_sorted(:,xi);
cellid_sorted = cellid_sorted((xi));
sample_sorted = sample_sorted((xi));
mapped_xy = mapped_xy(xi,:);
side_flag_sorted = side_flag_sorted(xi);
male_flag_sorted = male_flag_sorted(xi);
female_flag_sorted = female_flag_sorted(xi);
parent_flag_sorted = parent_flag_sorted(xi);
cntnp_flag_sorted = cntnp_flag_sorted(xi);
prj_sorted = prj_sorted(xi,:);

meangr_mat = zeros(length(moldata(:,1)),length(idxuni));
clust_cent = zeros(length(idxuni),2);
for jjj=1:length(idxuni)
    jjj
    meangr_mat(:,jjj) = mean(log2(data_sorted_all(:,idx==idxuni(jjj))+1),2);
    clust_cent(jjj,:) = [median(mapped_xy(idx==idxuni(jjj),1)),median(mapped_xy(idx==idxuni(jjj),2))];
end
meangr_mat1 = meangr_mat;
% meangr_mat1(loc,:) = [];

% meangr_mat1 = cent_norm(meangr_mat1(:,leaforder));
[prj,m,D,V,Q] = pca_wis(meangr_mat1',initial_dims);
Zpca = linkage(prj,'ward','correlation');
Dpca = pdist(prj,'correlation');
leaforder_pca = optimalleaforder(Zpca,Dpca);
figure;
set(gcf,'position',[100,100,1000,1000],'color','w')
axes('position',[0.03,0.03,0.3,0.93])
% hden = dendrogram(Zpca,length(leaforder_pca),'Orientation','left');
hden = dendrogram(Zpca,length(leaforder_pca),'Reorder',leaforder_pca,'Orientation','left');
axis off
set(gca,'ylim',[0.5,length(leaforder_pca)+0.5])
axes('position',[0.35,0.03,0.63,0.93])
x=squareform(Dpca); imagesc(x(leaforder_pca,leaforder_pca));
colormap('summer')
set(gca,'ytick',[1:length(leaforder_pca)],'xtick',[],'fontsize',8,'ydir','normal')

if savefig_flag==1
    savefig(gcf,['tree_QC2000_',date,'.fig'])    
end
leaforder = leaforder_pca;


T_cells_tmp_new = zeros(size(idx));
for i=1:length(leaforder)
    T_cells_tmp_new(idx==idxuni(leaforder(i))) = i;
end
idxuni_new = unique(T_cells_tmp_new);
[~,xi] = sort(T_cells_tmp_new);
T_cells_tmp_new = T_cells_tmp_new(xi);
data_sorted_all = data_sorted_all(:,xi);
data_orig_all_sorted = data_orig_all_sorted(:,xi);
cellid_sorted = cellid_sorted(xi);
% % tissue = tissue(xi);
mapped_xy = mapped_xy(xi,:);
side_flag_sorted = side_flag_sorted(xi);
male_flag_sorted = male_flag_sorted(xi);
female_flag_sorted = female_flag_sorted(xi);
parent_flag_sorted = parent_flag_sorted(xi);
cntnp_flag_sorted = cntnp_flag_sorted(xi);
cells_bor_2 = find(diff(T_cells_tmp_new)>0)+1;
sample_sorted = sample_sorted(xi);
prj_sorted = prj_sorted(xi,:);
T_cells_tmp = T_cells_tmp_new;
T_cells_tmp_uni = unique(T_cells_tmp);
% % % %
idx = T_cells_tmp_new;
idxuni = idxuni_new;

colors = distinguishable_colors(length(T_cells_tmp_uni)+1);
figure;
set(gcf,'color','w','position',[20,20,900,800])
for i=1:length(T_cells_tmp_uni)
    if i==-1
        ii=find(T_cells_tmp==i); h=plot(mapped_xy(ii,1),mapped_xy(ii,2),'.','color',[0,0,1],'markersize',3); hold on;
    else
        ii=find(T_cells_tmp==i); h=plot(mapped_xy(ii,1),mapped_xy(ii,2),'.','color',colors(i+1,:),'markersize',5); hold on;
    end
end
for i=1:length(T_cells_tmp_uni)
    in = T_cells_tmp==i;
    ht = text(median(mapped_xy(in,1)),median(mapped_xy(in,2)),num2str(i));
    set(ht,'BackgroundColor',0.8*[1,1,1],'fontsize',8)
end
axis tight;
axis equal
axis off
title(['MinPts=',num2str(MinPts),', epsprc=',num2str(eps_prc),',#C=',num2str(max(T_cells_tmp)),',#out=',num2str(sum(T_cells_tmp==0))],'fontsize',8);
if savefig_flag==1
    savefig(gcf,['tsne_only_neurons_by_cluster_v1_',date,'.fig'])
    % eval(['export_fig tsne_AmyPiri_FC_perplexity_',num2str(perplexity),'_Ngenes=',num2str(length(data_tsne(:,1))),'_NPCA=',num2str(initial_dims),'_',date,'.pdf']);
end
if savefig_pdf==1
    eval(['export_fig tsne_only_neurons_by_cluster_v1_',date,'.pdf']);
end
% % % % % % % % % % % % % 
sample_uni = {'35-1','35-2','36-1','36-2','37-1','37-2','38-1','38-2','51-1','51-2','51-3','51-4','52-2','52-3','52-4','54-1','54-2'};
colors = distinguishable_colors(length(sample_uni)+1);
figure;
set(gcf,'color','w','position',[20,20,1200,1200]);
[ha, pos] = tight_subplot(2, 2, [0.02,0.02], [0.02,0.02], [0.02,0.02]);
axes(ha(1))
for i=1:length(sample_uni)
    s = scatter(mapped_xy(strcmpi(sample_sorted,sample_uni{i}),1),mapped_xy(strcmpi(sample_sorted,sample_uni{i}),2),10,colors(i,:),'filled'); hold on;
    alpha(s,0.4);
end
axis tight
axis equal
axis off
legend({'35-1,F,L','35-2,F,R','36-1,M,L','36-2,M,R'...
    ,'37-1,M,L','37-2,M,R','38-1,F,L','38-2,F,R',....
    '51-1,F','51-2,F','51-3,M','51-4,M','52-1,F','52-2,F','52-3,M','52-4,M','54-1,F','54-2,F'})
% plot by left/right
side_flag_uni = [1,-1,0];
colors = [1,0,0;0,0,1,;0,1,0]; %distinguishable_colors(length(sample_uni)+1);
axes(ha(2))
for i=1:length(side_flag_uni)
    s = scatter(mapped_xy(side_flag_sorted==side_flag_uni(i),1),mapped_xy(side_flag_sorted==side_flag_uni(i),2),10,colors(i,:),'filled'); hold on;
    alpha(s,0.3);
end
axis tight
axis equal
axis off
legend('Left','Right','Mixed')
% plot by male/female
colors = [1,0,0;0,0,1;1,0,0;0,0,1]; %distinguishable_colors(length(sample_uni)+1);
axes(ha(3))
s = scatter(mapped_xy(male_flag_sorted,1),mapped_xy(male_flag_sorted,2),10,colors(1,:),'filled'); hold on;
alpha(s,0.3);
s = scatter(mapped_xy(female_flag_sorted,1),mapped_xy(female_flag_sorted,2),10,colors(2,:),'filled'); hold on;
alpha(s,0.3);
axis tight
axis off
axis equal
legend('Male','Female')
axes(ha(4))
s = scatter(mapped_xy(parent_flag_sorted==0,1),mapped_xy(parent_flag_sorted==0,2),10,colors(1,:),'filled'); hold on;
alpha(s,0.3);
s = scatter(mapped_xy(parent_flag_sorted,1),mapped_xy(parent_flag_sorted,2),10,colors(2,:),'filled'); hold on;
alpha(s,0.3);
axis tight
axis off
axis equal
legend('Virgin','Parent')
if savefig_flag==1
    savefig(gcf,['tsne_only_neurons_by_annot_v1_perplexity',num2str(perplexity),'_',date,'.fig'])
%     savefig(gcf,['umap_only_neurons_by_annot_v1_Nneighbors_',num2str(n_neighbors),'_minDist',num2str(min_dist),'_',date,'.fig'])
end
% % % % % % % % % % % % % 
[ind_gr_tmp_mark,cells_bor,gr_center] = markertablefeatures(T_cells_tmp,data_sorted_all,5);
% % % % % % % % % 
datamarkers = data_sorted_all(ind_gr_tmp_mark,:);
datamarkers_cn = cent_norm(log2(datamarkers+1));
% gr_tmp_mark = gr_tmp_mark(xi);
gr_tmp_mark = geneid(ind_gr_tmp_mark);

figure;
set(gcf,'position',[100,100,1400,770],'color','w')
ax1 = axes('position',[0.1,0.02,0.88,0.84]);
imagesc(datamarkers_cn,[prctile(datamarkers_cn(:),1),prctile(datamarkers_cn(:),99)]);
hold on;
linewid =0.5;
bor_color = 'grey11';%'green1';%
for jj=1:length(cells_bor)
    plot(cells_bor(jj)*[1,1]-0.5,[1,length(gr_tmp_mark)],'-','linewidth',linewid,'color',get_RGB(bor_color))
end
set(gca,'xtick',gr_center,'xticklabel',[1:length(gr_center)],'ytick',[1:length(gr_tmp_mark)],'yticklabel',gr_tmp_mark, 'fontsize', 10)
colormap('summer');
freezeColors(gca);

sample_uni = {'35-1','35-2','36-1','36-2','37-1','37-2','38-1','38-2','51-1','51-2','51-3','51-4','52-2','52-3','52-4','54-1','54-2'};
samples_num = false(length(sample_uni),length(sample_sorted));
for i=1:length(sample_uni)
    samples_num(i, strcmpi(sample_sorted,sample_uni{i})) = true;
end

ax2 = axes('position',[0.1,0.86,0.88,0.01]);
imagesc(side_flag_sorted'); hold on;
colormap('gray');
freezeColors(gca);
set(gca,'xtick',[],'ytick',[1],'yticklabel','Left');

ax3 = axes('position',[0.1,0.87,0.88,0.01]);
imagesc(~male_flag_sorted'); hold on;
colormap('gray');
freezeColors(gca);
set(gca,'xtick',[],'ytick',[1],'yticklabel','Male');

ax4 = axes('position',[0.1,0.88,0.88,0.01]);
imagesc(~parent_flag_sorted'); hold on;
colormap('gray');
freezeColors(gca);
set(gca,'xtick',[],'ytick',[1],'yticklabel','Parent');

ax5 = axes('position',[0.1,0.892,0.88,0.07]);
imagesc(~samples_num); hold on;
grid on
colormap('gray');
freezeColors(gca);
set(gca,'xtick',[],'ytick',[1:length(sample_uni)],'yticklabel',sample_uni,'fontsize',6);

gad2 = data_orig_all_sorted(strcmpi(geneid_all,'Gad2'),:);
slc17a7 = data_orig_all_sorted(strcmpi(geneid_all,'Slc17a7'),:);
slc17a6 = data_orig_all_sorted(strcmpi(geneid_all,'Slc17a6'),:);

ax6 = axes('position',[0.1,0.965,0.88,0.01]);
imagesc(~gad2); hold on;
axes(ax6)
colormap('gray');
freezeColors(ax6);
set(ax6,'xtick',[],'ytick',[1],'yticklabel','Gad2','fontsize',5);
ax7 = axes('position',[0.1,0.965+0.01,0.88,0.01]);
imagesc(~slc17a6); hold on;
colormap('gray');
freezeColors(ax7);
set(gca,'xtick',[],'ytick',[1],'yticklabel','Slc17a6','fontsize',6);
ax8 = axes('position',[0.1,0.965+2*0.01,0.88,0.01]);
imagesc(~slc17a7); hold on;
colormap('gray');
freezeColors(ax8);
set(gca,'xtick',[],'ytick',[1],'yticklabel','Slc17a7','fontsize',6);
linkaxes([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8],'x');
if savefig_flag==1
    savefig(gcf,['markertable_only_neurons_v1_',date,'.fig'])
    % eval(['export_fig markertable_AmyPiri_FC_',date,'.pdf']);
end
if savefig_pdf==1
    eval(['export_fig markertable_only_neurons_v1_',date,'.pdf']);
end
% % % % % % % % % % % % % % % % % % % % % % % % % % % 
% exclude_markers = {'C1qc','C1qa','C1qb','Gja1','Cx3cr1','Acta2','Ly6c1','Mfge8','Plp1'....
%     ,'Aqp4','Vtn','Cldn5','Pdgfrb','Flt1','Slc1a3','Pdgfra','Foxj1','Olig1','Olig2','Sox10','Hbb-bs','Hbb-bt','Hba-a2'};
% exclude_markers = {'C1qc','C1qa','C1qb','Gja1','Cx3cr1','Acta2','Ly6c1','Mfge8','Plxnb3','Cldn11'....
%     ,'Aqp4','Vtn','Cldn5','Pdgfrb','Flt1','Slc25a18','Pdgfra','Foxj1','Olig1','Olig2','Sox10','Hbb-bs','Hbb-bt','Hba-a2','Ttr'};

[~,loc] = ismember(exclude_markers,geneid_all);
nonneuro = sum(data_orig_all_sorted(loc,:));

gabaglut = zeros(size(T_cells_tmp_uni));
gabaglut_sc = zeros(size(T_cells_tmp));
for jjj=1:length(T_cells_tmp_uni)
    jjj
    %     tmp = [mean(gad2(:,T_cells_tmp==jjj)>0);mean(slc17a7(:,T_cells_tmp==jjj)>0);....
    %         mean(slc17a6(:,T_cells_tmp==jjj)>0);mean(nonneuro(:,T_cells_tmp==jjj)>0)];
    tmp = [mean(gad2(:,T_cells_tmp==jjj));mean(slc17a7(:,T_cells_tmp==jjj));....
        mean(slc17a6(:,T_cells_tmp==jjj));mean(nonneuro(:,T_cells_tmp==jjj))];
    tmpsort = [tmp(1),max(tmp(2:3)),tmp(4)];
    tmpsort= sort(tmpsort,'descend');
    if tmpsort(1)>2*tmpsort(2)
        [~,gabaglut(jjj)] = max(tmp);
    else
        gabaglut(jjj) = 5;
    end
    gabaglut_sc(T_cells_tmp==jjj) = gabaglut(jjj);
end
figure;
set(gcf,'color','w','position',[20,20,800,800])
colors = distinguishable_colors(length(unique(gabaglut_sc)));
for idx=unique(gabaglut_sc)'
    ii=find(gabaglut_sc==idx); 
    h=plot(mapped_xy(ii,1),mapped_xy(ii,2),'.','color',colors(idx,:),'markersize',3); hold on;
end
axis tight
axis equal
axis off
title('GABA/Glut1/Glut2')
legend('GABA','Glut1','Glut2','non-neurons','doublets')
% cid = cellfun(@(x,y) [x,'_',y], cellid_sorted, sample_sorted,'UniformOutput',0);
figure('color','w','position',[20,20,800,800])
markergene = nonneuro;
inpos = markergene>0;
tmpthlow = prctile(markergene(markergene>0),1);
tmpthhigh = prctile(markergene(markergene>0),90);
markergene(markergene>tmpthhigh) = tmpthhigh;
markergene(markergene<tmpthlow) = tmpthlow;
c_rgb = [1,0,0];rand([1,3]);
markergene_color = [interp1([min(markergene),max(markergene)],[0.7,c_rgb(1)],markergene'),...
    interp1([min(markergene),max(markergene)],[0.7,c_rgb(2)],markergene')...
    ,interp1([min(markergene),max(markergene)],[0.7,c_rgb(3)],markergene')];
scatter(mapped_xy(~inpos,1),mapped_xy(~inpos,2),20,markergene_color(~inpos,:),'.'); hold on;
scatter(mapped_xy(inpos,1),mapped_xy(inpos,2),20,markergene_color(inpos,:),'.'); hold on;
title('nonneurons sum');
axis tight
axis equal
axis off

table1 = [cellid_sorted,m2c(gabaglut_sc)];
saveCellFile(table1,['cellid_gaba_glut_QC3000_Sdimor',date,'.txt']);
% % % % % % % % % % % % %
list = {'Snap25','Stmn2','Gad2','Slc32a1','Slc17a7','Slc17a6','Sst','Sim1','Tac2','Ttr','Foxj1','Acta2','Flt1','Cldn5','Aqp4','Plp1'};
figure;
set(gcf,'color','w','position',[20,20,1100,960])
[ha, pos] = tight_subplot(4, 4, [0.02,0.02], [0.02,0.02], [0.02,0.02]);
for i=1:length(list)
    genePlot = list{i};
    markergene = (data_orig_all_sorted(strcmpi(geneid_all,genePlot),:));
    inpos = markergene>0;
    tmpthlow = prctile(markergene(markergene>0),1);
    tmpthhigh = prctile(markergene(markergene>0),99);
    if tmpthlow==tmpthhigh
        tmpthlow = 0;
    end
    markergene(markergene>tmpthhigh) = tmpthhigh;
    markergene(markergene<tmpthlow) = tmpthlow;
    c_rgb = [1,0,0];rand([1,3]);
    markergene_color = [interp1([min(markergene),max(markergene)],[0.7,c_rgb(1)],markergene'),...
        interp1([min(markergene),max(markergene)],[0.7,c_rgb(2)],markergene')...
        ,interp1([min(markergene),max(markergene)],[0.7,c_rgb(3)],markergene')];
    axes(ha(i));
    scatter(mapped_xy(~inpos,1),mapped_xy(~inpos,2),20,markergene_color(~inpos,:),'.'); hold on;
    scatter(mapped_xy(inpos,1),mapped_xy(inpos,2),20,markergene_color(inpos,:),'.'); hold on;
    set(gca,'xlim',[-150,150],'ylim',[-150,150])
    title(genePlot);
    axis tight
    axis equal
    axis off
end
if savefig_flag==1
    % eval(['export_fig tsne_markers2_AmyPiri_FC_',date,'.pdf']);
    savefig(gcf,['tsne_all_Markers1_neurons_v1_',date,'.fig'])
end
if savefig_pdf==1
    eval(['export_fig tsne_all_Markers1_neurons_v1_',date,'.pdf']);
end
% % % % % % % % % % % % %
list = {'Snap25','Stmn2','Gad2','Slc32a1','Slc17a7','Slc17a6','Sst','Pvalb','Vip','Htr3a','Gpr88','Tekt5','Reln','C1ql3','Nr2f2','Npas4'};
figure;
set(gcf,'color','w','position',[20,20,1100,960])
[ha, pos] = tight_subplot(4, 4, [0.02,0.02], [0.02,0.02], [0.02,0.02]);
for i=1:length(list)
    genePlot = list{i};
    markergene = (data_orig_all_sorted(strcmpi(geneid_all,genePlot),:));
    inpos = markergene>0;
    tmpthlow = prctile(markergene(markergene>0),1);
    tmpthhigh = prctile(markergene(markergene>0),99);
    if tmpthlow==tmpthhigh
        tmpthlow = 0;
    end
    markergene(markergene>tmpthhigh) = tmpthhigh;
    markergene(markergene<tmpthlow) = tmpthlow;
    c_rgb = [1,0,0];rand([1,3]);
    %     markergene_color = [interp1([min(markergene),max(markergene)],[0,1],markergene'),zeros(size(markergene'))...
    %         ,interp1([min(markergene),max(markergene)],[1,0],markergene')];
    markergene_color = [interp1([min(markergene),max(markergene)],[0.7,c_rgb(1)],markergene'),...
        interp1([min(markergene),max(markergene)],[0.7,c_rgb(2)],markergene')...
        ,interp1([min(markergene),max(markergene)],[0.7,c_rgb(3)],markergene')];
    axes(ha(i));
    scatter(mapped_xy(~inpos,1),mapped_xy(~inpos,2),20,markergene_color(~inpos,:),'.'); hold on;
    scatter(mapped_xy(inpos,1),mapped_xy(inpos,2),20,markergene_color(inpos,:),'.'); hold on;
    set(gca,'xlim',[-150,150],'ylim',[-150,150])
    title(genePlot);
    axis tight
    axis equal
    axis off
end
if savefig_flag==1
    % eval(['export_fig tsne_markers3_AmyPiri_FC_',date,'.pdf']);
    savefig(gcf,['tsne_all_Markers2_neurons_v1_',date,'.fig'])
end
if savefig_pdf==1
    eval(['export_fig tsne_all_Markers2_neurons_v1_',date,'.pdf']);
end
% % % % % % % % % % % % %
list = {'Slc32a1','Slc17a7','Egr1','Egr2','Egr4','Npas4','Arc','Jun','Junb','Fos','Fosb','Gadd45g','Btg2','Rrad','Nptx2','Peg10'};
figure;
set(gcf,'color','w','position',[20,20,1100,960])
[ha, pos] = tight_subplot(4, 4, [0.02,0.02], [0.02,0.02], [0.02,0.02]);
for i=1:length(list)
    genePlot = list{i};
    markergene = (data_orig_all_sorted(strcmpi(geneid_all,genePlot),:));
    inpos = markergene>0;
    tmpthlow = prctile(markergene(markergene>0),1);
    tmpthhigh = prctile(markergene(markergene>0),99);
    if tmpthlow==tmpthhigh
        tmpthlow = 0;
    end
    markergene(markergene>tmpthhigh) = tmpthhigh;
    markergene(markergene<tmpthlow) = tmpthlow;
    c_rgb = [1,0,0];rand([1,3]);
    %     markergene_color = [interp1([min(markergene),max(markergene)],[0,1],markergene'),zeros(size(markergene'))...
    %         ,interp1([min(markergene),max(markergene)],[1,0],markergene')];
    markergene_color = [interp1([min(markergene),max(markergene)],[0.7,c_rgb(1)],markergene'),...
        interp1([min(markergene),max(markergene)],[0.7,c_rgb(2)],markergene')...
        ,interp1([min(markergene),max(markergene)],[0.7,c_rgb(3)],markergene')];
    axes(ha(i));
    scatter(mapped_xy(~inpos,1),mapped_xy(~inpos,2),20,markergene_color(~inpos,:),'.'); hold on;
    scatter(mapped_xy(inpos,1),mapped_xy(inpos,2),20,markergene_color(inpos,:),'.'); hold on;
    set(gca,'xlim',[-150,150],'ylim',[-150,150])
    title(genePlot);
    axis tight
    axis equal
    axis off
end
if savefig_flag==1
    % eval(['export_fig tsne_markers3_AmyPiri_FC_',date,'.pdf']);
    savefig(gcf,['tsne_all_Markers3_neurons_v1_',date,'.fig'])
end
if savefig_pdf==1
    eval(['export_fig tsne_all_Markers3_neurons_v1_',date,'.pdf']);
end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % %
list = {'Slc32a1','Slc17a6','Gadd45b','Tmem198','Foxp2','Npas4','Arc','Vgf','Fndc9','Nr4a3','Fosb','Tshz1','Cxcl14','Rrad','Prox1','Gad2'};
figure;
set(gcf,'color','w','position',[20,20,1100,960])
[ha, pos] = tight_subplot(4, 4, [0.02,0.02], [0.02,0.02], [0.02,0.02]);
for i=1:length(list)
    genePlot = list{i};
    markergene = (data_orig_all_sorted(strcmpi(geneid_all,genePlot),:));
    inpos = markergene>0;
    tmpthlow = prctile(markergene(markergene>0),1);
    tmpthhigh = prctile(markergene(markergene>0),90);
    if tmpthlow==tmpthhigh
        tmpthlow = 0;
    end
    markergene(markergene>tmpthhigh) = tmpthhigh;
    markergene(markergene<tmpthlow) = tmpthlow;
    c_rgb = [1,0,0];rand([1,3]);
    %     markergene_color = [interp1([min(markergene),max(markergene)],[0,1],markergene'),zeros(size(markergene'))...
    %         ,interp1([min(markergene),max(markergene)],[1,0],markergene')];
    markergene_color = [interp1([min(markergene),max(markergene)],[0.7,c_rgb(1)],markergene'),...
        interp1([min(markergene),max(markergene)],[0.7,c_rgb(2)],markergene')...
        ,interp1([min(markergene),max(markergene)],[0.7,c_rgb(3)],markergene')];
    axes(ha(i));
    scatter(mapped_xy(~inpos,1),mapped_xy(~inpos,2),20,markergene_color(~inpos,:),'.'); hold on;
    scatter(mapped_xy(inpos,1),mapped_xy(inpos,2),20,markergene_color(inpos,:),'.'); hold on;
    set(gca,'xlim',[-150,150],'ylim',[-150,150])
    title(genePlot);
    axis tight
    axis equal
    axis off
end
if savefig_flag==1
    % eval(['export_fig tsne_markers3_AmyPiri_FC_',date,'.pdf']);
    savefig(gcf,['tsne_all_Markers4_neurons_v1_',date,'.fig'])
end
if savefig_pdf==1
    eval(['export_fig tsne_all_Markers3_neurons_v1_',date,'.pdf']);
end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
list = {'Slc32a1','Slc17a6','Sim1','Pou3f1','Prdm8','Dach1','Cyp26b1','Cbln2','Qrfpr','C1ql1','Synpr','Prss23','Lypd1','Trh','Mc4r','Otp'};
figure;
set(gcf,'color','w','position',[20,20,1100,960])
[ha, pos] = tight_subplot(4, 4, [0.02,0.02], [0.02,0.02], [0.02,0.02]);
for i=1:length(list)
    genePlot = list{i};
    markergene = (data_orig_all_sorted(strcmpi(geneid_all,genePlot),:));
    inpos = markergene>0;
    tmpthlow = prctile(markergene(markergene>0),1);%0;%
    tmpthhigh = prctile(markergene(markergene>0),99);%1;%
    if tmpthlow==tmpthhigh
        tmpthlow = 0;
    end
    markergene(markergene>tmpthhigh) = tmpthhigh;
    markergene(markergene<tmpthlow) = tmpthlow;
    c_rgb = [1,0,0];rand([1,3]);
    %     markergene_color = [interp1([min(markergene),max(markergene)],[0,1],markergene'),zeros(size(markergene'))...
    %         ,interp1([min(markergene),max(markergene)],[1,0],markergene')];
    markergene_color = [interp1([min(markergene),max(markergene)],[0.7,c_rgb(1)],markergene'),...
        interp1([min(markergene),max(markergene)],[0.7,c_rgb(2)],markergene')...
        ,interp1([min(markergene),max(markergene)],[0.7,c_rgb(3)],markergene')];
    axes(ha(i));
    scatter(mapped_xy(~inpos,1),mapped_xy(~inpos,2),20,markergene_color(~inpos,:),'.'); hold on;
    scatter(mapped_xy(inpos,1),mapped_xy(inpos,2),20,markergene_color(inpos,:),'.'); hold on;
    set(gca,'xlim',[-150,150],'ylim',[-150,150])
    title(genePlot);
    axis tight
    axis equal
    axis off
end
if savefig_flag==1
    % eval(['export_fig tsne_markers3_AmyPiri_FC_',date,'.pdf']);
    savefig(gcf,['tsne_all_Markers5_neurons_v1_',date,'.fig'])
end
if savefig_pdf==1
    eval(['export_fig tsne_all_Markers5_neurons_v1_',date,'.pdf']);
end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
list = {'Slc32a1','Slc17a6','Avp','Lhx8','Ecel1','Slc5a7','Slc18a3','Prlr','Cox6a2','Sim1','Cnr1','Cxcl14','Scg2','Col12a1','Adcyap1','Slc17a8'};
figure;
set(gcf,'color','w','position',[20,20,1100,960])
[ha, pos] = tight_subplot(4, 4, [0.02,0.02], [0.02,0.02], [0.02,0.02]);
for i=1:length(list)
    genePlot = list{i};
    markergene = (data_orig_all_sorted(strcmpi(geneid_all,genePlot),:));
    inpos = markergene>0;
    tmpthlow = prctile(markergene(markergene>0),1);%0;%
    tmpthhigh = prctile(markergene(markergene>0),95);%1;%
    if tmpthlow==tmpthhigh
        tmpthlow = 0;
    end
    markergene(markergene>tmpthhigh) = tmpthhigh;
    markergene(markergene<tmpthlow) = tmpthlow;
    c_rgb = [1,0,0];rand([1,3]);
    %     markergene_color = [interp1([min(markergene),max(markergene)],[0,1],markergene'),zeros(size(markergene'))...
    %         ,interp1([min(markergene),max(markergene)],[1,0],markergene')];
    markergene_color = [interp1([min(markergene),max(markergene)],[0.7,c_rgb(1)],markergene'),...
        interp1([min(markergene),max(markergene)],[0.7,c_rgb(2)],markergene')...
        ,interp1([min(markergene),max(markergene)],[0.7,c_rgb(3)],markergene')];
    axes(ha(i));
    scatter(mapped_xy(~inpos,1),mapped_xy(~inpos,2),20,markergene_color(~inpos,:),'.'); hold on;
    scatter(mapped_xy(inpos,1),mapped_xy(inpos,2),20,markergene_color(inpos,:),'.'); hold on;
    set(gca,'xlim',[-150,150],'ylim',[-150,150])
    title(genePlot);
    axis tight
    axis equal
    axis off
end
if savefig_flag==1
    % eval(['export_fig tsne_markers3_AmyPiri_FC_',date,'.pdf']);
    savefig(gcf,['tsne_all_Markers6_neurons_v1_',date,'.fig'])
end
if savefig_pdf==1
    eval(['export_fig tsne_all_Markers7_neurons_v1_',date,'.pdf']);
end
% % % % % % % % % % % % % % % % % % % % % % % % % %
list = {'Slc32a1','Slc17a6','Adora2a','Penk','Drd1','Ido1','Rgs9','Ppp1r1b','Mhrt','Gpr83','Gpr88','Tac1','Foxp2','Rxrg','Wnt2','Isl1'};
figure;
set(gcf,'color','w','position',[20,20,1100,960])
[ha, pos] = tight_subplot(4, 4, [0.02,0.02], [0.02,0.02], [0.02,0.02]);
for i=1:length(list)
    genePlot = list{i};
    markergene = (data_orig_all_sorted(strcmpi(geneid_all,genePlot),:));
    inpos = markergene>0;
    tmpthlow = prctile(markergene(markergene>0),1);%0;%
    tmpthhigh = prctile(markergene(markergene>0),95);%1;%
    if tmpthlow==tmpthhigh
        tmpthlow = 0;
    end
    markergene(markergene>tmpthhigh) = tmpthhigh;
    markergene(markergene<tmpthlow) = tmpthlow;
    c_rgb = [1,0,0];rand([1,3]);
    %     markergene_color = [interp1([min(markergene),max(markergene)],[0,1],markergene'),zeros(size(markergene'))...
    %         ,interp1([min(markergene),max(markergene)],[1,0],markergene')];
    markergene_color = [interp1([min(markergene),max(markergene)],[0.7,c_rgb(1)],markergene'),...
        interp1([min(markergene),max(markergene)],[0.7,c_rgb(2)],markergene')...
        ,interp1([min(markergene),max(markergene)],[0.7,c_rgb(3)],markergene')];
    axes(ha(i));
    scatter(mapped_xy(~inpos,1),mapped_xy(~inpos,2),20,markergene_color(~inpos,:),'.'); hold on;
    scatter(mapped_xy(inpos,1),mapped_xy(inpos,2),20,markergene_color(inpos,:),'.'); hold on;
    set(gca,'xlim',[-150,150],'ylim',[-150,150])
    title(genePlot);
    axis tight
    axis equal
    axis off
end
if savefig_flag==1
    % eval(['export_fig tsne_markers3_AmyPiri_FC_',date,'.pdf']);
    savefig(gcf,['tsne_all_Markers6_neurons_v1_',date,'.fig'])
end
if savefig_pdf==1
    eval(['export_fig tsne_all_Markers8_neurons_v1_',date,'.pdf']);
end
% % % % % % % % % % % % % % % % % % % % % % % % % %
marker = data_orig_all_sorted(strcmpi(geneid_all,'Fos'),:);
marker_percent = zeros(1, length(unique(idx)));
for j=1:length(idxuni)
    c1 = sum( idx==idxuni(j));
    c2 = sum( idx==idxuni(j) & marker'>0);
    marker_percent(j) = 100*c2/c1;
%     marker_percent(j) = mean(marker(idx==idxuni(j)));
end

marker_percent(isnan(marker_percent)) = 0;
fos_percent = marker_percent;

marker = data_orig_all_sorted(strcmpi(geneid_all,'Arc'),:);
marker_percent = zeros(1, length(unique(idx)));
for j=1:length(idxuni)
    c1 = sum( idx==idxuni(j));
    c2 = sum( idx==idxuni(j) & marker'>0);
    marker_percent(j) = 100*c2/c1;
%     marker_percent(j) = mean(marker(idx==idxuni(j)));
end
marker_percent(isnan(marker_percent)) = 0;
arc_percent = marker_percent;

marker = data_orig_all_sorted(strcmpi(geneid_all,'Npas4'),:);
marker_percent = zeros(1, length(unique(idx)));
for j=1:length(idxuni)
    c1 = sum( idx==idxuni(j));
    c2 = sum( idx==idxuni(j) & marker'>0);
    marker_percent(j) = 100*c2/c1;
%     marker_percent(j) = mean(marker(idx==idxuni(j)));
end
marker_percent(isnan(marker_percent)) = 0;
npas4_percent = marker_percent;

M = length(idx);
left_male = mean(side_flag_sorted(male_flag_sorted));
left_female = mean(side_flag_sorted(~male_flag_sorted));
K = sum(side_flag_sorted);
marker = side_flag_sorted';
marker_percent = zeros(1, length(unique(idx)));
p_left = zeros(1, length(unique(idx)));
for j=1:length(idxuni)
    c1 = sum( idx==idxuni(j));
    c1_amy = sum( idx(male_flag_sorted)==idxuni(j));
    c1_piri = sum( idx(~male_flag_sorted)==idxuni(j));
    c2 = sum( idx==idxuni(j) & marker'>0);
    K = round((c1_amy*left_male + c1_piri*left_female)/c1*M);
    marker_percent(j) = 100*c2/c1;
    p_left(j) = hygecdf(c2,M,K,c1,'upper');
end
marker_percent(isnan(marker_percent)) = 0;
left_percent = marker_percent;

M = length(idx);
K = sum(male_flag_sorted);
marker = male_flag_sorted';
marker_percent = zeros(1, length(unique(idx)));
p_male = zeros(1, length(unique(idx)));
for j=1:length(idxuni)
    c1 = sum( idx==idxuni(j));
    c2 = sum( idx==idxuni(j) & marker'>0);
    marker_percent(j) = 100*c2/c1;
    p_male(j) = hygecdf(c2,M,K,c1,'upper');
end
marker_percent(isnan(marker_percent)) = 0;
male_percent = marker_percent;

p_female = 1-p_male;
p_male(p_male<1e-10) = 1e-10;
p_female(p_female<1e-10) = 1e-10;
p_left(p_left<1e-10) = 1e-10;


t = [fos_percent;arc_percent;npas4_percent];

figure;
set(gcf,'color','w','position',[20,20,900,1200])
[ha, pos] = tight_subplot(1, 3, [0.02,0.00], [0.02,0.02], [0.05,0.02]);
axes(ha(1));
barh(t');
set(gca,'ytick',[1:length(idxuni)])
legend({'Fos','Arc','Npas4','Left','Male'});

t = [left_percent;male_percent];
axes(ha(2));
barh(t');
set(gca,'ytick',[1:length(idxuni)])

legend({'Left','Male'});

t = -log10([p_left;p_male;(1-p_male)]);
t(t>10) = 10;
axes(ha(3));
barh(t');
set(gca,'ytick',[1:length(idxuni)])
legend({'Left','Male','Female'});


linkaxes([ha(1),ha(2),ha(3)],'y');
if savefig_flag==1
    savefig(gcf,['immidiateEarly_enrichment_byCluster_all_neurons_v1_',date,'.fig'])
end
if savefig_pdf==1
    eval(['export_fig immidiateEarly_enrichment_byCluster_all_neurons_v1_',date,'.pdf']);
end
% % % % % % % % % % % % % % % % % % % % % % % % %
c1 = 50; c2 = 89;
top_g = 20;
gr1 = find(idx==c1 );
gr2 = find(idx==c2 );
x1 = mean(log2(data_sorted_all(:,gr1)+1),2);
x2 = mean(log2(data_sorted_all(:,gr2)+1),2);
d = x1-x2 ;
[~,xi] = sort(d);
figure('position',[200,200,1000,580],'color','w');
[ha, pos] = tight_subplot(1, 2, [0.05,0.05], [0.1,0.05], [0.05,0.05]);
axes(ha(1))
plot(x1, x2, '.');hold on;
xmax = max(x1);
plot([0,xmax],[0,xmax],'-k'); grid on
plot([0,xmax],[0,xmax]+1,'--k'); grid on
plot([1,xmax],[0,xmax-1],'--k'); grid on
plot(x1(xi(1:top_g)), x2(xi(1:top_g)),'.r'); hold on;
text(x1(xi(1:top_g)), x2(xi(1:top_g)),geneid(xi(1:top_g)),'fontsize',6);
xi = flipud(xi);
plot(x1(xi(1:top_g)), x2(xi(1:top_g)),'.r'); hold on;
text(x1(xi(1:top_g)), x2(xi(1:top_g)),geneid(xi(1:top_g)),'fontsize',6);
title(['cluster ',num2str(c2),' vs ',num2str(c1)])
xlabel(['mean (',num2str(c1),')'])
ylabel(['mean (',num2str(c2),')'])
axis tight

top_g = 20;
gr1 = find(idx==c1 );
gr2 = find(idx==c2 );
x1 = mean(data_sorted_all(:,gr1)>0,2);
x2 = mean(data_sorted_all(:,gr2)>0,2);
d = x1-x2 ;
[~,xi] = sort(d);
axes(ha(2))
plot(x1, x2, '.');hold on;
xmax = max(x1);
plot([0,xmax],[0,xmax],'-k'); grid on
plot([0,xmax],[0,xmax]+0.4,'--k'); grid on
plot([0.4,xmax],[0,xmax-0.4],'--k'); grid on
plot(x1(xi(1:top_g)), x2(xi(1:top_g)),'.r'); hold on;
text(x1(xi(1:top_g)), x2(xi(1:top_g)),geneid(xi(1:top_g)),'fontsize',6);
xi = flipud(xi);
plot(x1(xi(1:top_g)), x2(xi(1:top_g)),'.r'); hold on;
text(x1(xi(1:top_g)), x2(xi(1:top_g)),geneid(xi(1:top_g)),'fontsize',6);
title(['cluster ',num2str(c2),' vs ',num2str(c1)])
xlabel(['mean (',num2str(c1),')'])
ylabel(['mean (',num2str(c2),')'])
axis tight
if savefig_flag==1
    savefig(gcf,['scatter_cluster_',num2str(c1),'_vs_',num2str(c2),'_',date,'.fig'])
end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

marker = data_orig_all_sorted(strcmpi(geneid_all,'Sim1'),:);
marker_percent = zeros(1, length(unique(idx)));
for j=1:length(idxuni)
    c1 = sum( idx==idxuni(j));
    c2 = sum( idx==idxuni(j) & marker'>0);
    marker_percent(j) = 100*c2/c1;
end
marker_percent(isnan(marker_percent)) = 0;
sim1_percent = marker_percent;
sim1_by_sample = 100*sum(samples_num(:,marker>0),2)./sum(samples_num,2);
sim1_fc = 100*sum(marker>0 & side_flag_sorted'>0)/sum(marker>0);
sim1_amy = 100*sum(marker>0 & male_flag_sorted'>0)/sum(marker>0);

marker = data_orig_all_sorted(strcmpi(geneid_all,'Rrad'),:);
marker_percent = zeros(1, length(unique(idx)));
for j=1:length(idxuni)
    c1 = sum( idx==idxuni(j));
    c2 = sum( idx==idxuni(j) & marker'>0);
    marker_percent(j) = 100*c2/c1;
end
marker_percent(isnan(marker_percent)) = 0;
rrad_percent = marker_percent;
rrad_by_sample = 100*sum(samples_num(:,marker>0),2)./sum(samples_num,2);
rrad_fc = 100*sum(marker>0 & side_flag_sorted'>0)/sum(marker>0);
rrad_amy = 100*sum(marker>0 & male_flag_sorted'>0)/sum(marker>0);


t = [rrad_percent;sim1_percent];

figure;
set(gcf,'color','w','position',[20,20,900,1200])
[ha, pos] = tight_subplot(1, 3, [0.02,0.05], [0.02,0.02], [0.05,0.02]);
axes(ha(1));
barh(t');
set(gca,'ytick',[1:length(idxuni)])
legend({'Rrad','Sim1'});

t = [rrad_by_sample';sim1_by_sample'];
axes(ha(2));
barh(t');
set(gca,'ytick',[1:length(sample_uni)],'YTickLabel',sample_uni)
legend({'Rrad','Sim1'});

t = [ [rrad_fc;sim1_fc;100*mean(side_flag_sorted)],[rrad_amy;sim1_amy;100*mean(male_flag_sorted)] ];
axes(ha(3));
barh(t');
set(gca,'ytick',[1:2],'YTickLabel',{'FC','Amy'})
legend({'Rrad','Sim1','ctrl'});
if savefig_pdf==1
    eval(['export_fig Sim1_Rrad_byCluster_all_neurons_v1_',date,'.pdf']);
end

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
sample_uni = {'8-1','8-2','10-1','10-2','18-1','18-2','19-1','23-1','23-2','23-3'};
colors = distinguishable_colors(length(sample_uni)+1);
figure;
set(gcf,'color','w','position',[20,20,1800,800]);
[ha, pos] = tight_subplot(1, 3, [0.02,0.02], [0.02,0.02], [0.02,0.02]);
axes(ha(1))
for i=1:length(sample_uni)
    s = scatter(mapped_xy(strcmpi(sample_sorted,sample_uni{i}),1),mapped_xy(strcmpi(sample_sorted,sample_uni{i}),2),10,colors(i,:),'filled'); hold on;
    alpha(s,0.4);
end
axis tight
axis equal
axis off
legend({'8-1,Amy ct','8-2,Piri ct','10-1,Amy FC','10-2,Piri FC'....
    ,'18-1,Amy FC','18-2,Piri FC','19-1,Amy FC','23-1,Amy ct','23-2,Piri ct','23-3,Amy ct'})
% plot by FC
side_flag_uni = [1,0];
colors = [1,0,0;0,0,1]; %distinguishable_colors(length(sample_uni)+1);
axes(ha(2))
for i=1:length(side_flag_uni)
    s = scatter(mapped_xy(side_flag_sorted==side_flag_uni(i),1),mapped_xy(side_flag_sorted==side_flag_uni(i),2),10,colors(i,:),'filled'); hold on;
    alpha(s,0.3);
end
axis tight
axis equal
axis off
legend('FC','ctrl')
% plot by tissue
colors = [1,0,0;0,0,1;1,0,0;0,0,1]; %distinguishable_colors(length(sample_uni)+1);
axes(ha(3))
s = scatter(mapped_xy(male_flag_sorted,1),mapped_xy(male_flag_sorted,2),10,colors(1,:),'filled'); hold on;
alpha(s,0.3);
s = scatter(mapped_xy(female_flag_sorted,1),mapped_xy(female_flag_sorted,2),10,colors(2,:),'filled'); hold on;
alpha(s,0.3);
axis tight
axis off
axis equal
legend('Amy','Piri')
if savefig_flag==1
    % eval(['export_fig tsne_AmyPiri_FC_by_sampleannot_','_',date,'.pdf']);
    savefig(gcf,['tsne_all_onNeurons_by_annot_v1_Nneighbors_',num2str(n_neighbors),'_minDist',num2str(min_dist),'_',date,'.fig'])
end
if savefig_pdf==1
    eval(['export_fig tsne_by_annot_all_neurons_v1_',date,'.pdf']);
end
% % % % % % % % % % % % % % % % % % % %
colors = distinguishable_colors(length(sample_uni)+1);
figure;
set(gcf,'color','w','position',[20,20,1800,800]);
[ha, pos] = tight_subplot(1, 3, [0.02,0.02], [0.02,0.02], [0.02,0.02]);
axes(ha(1))
side_flag_uni = [1,0];
colors = [1,0,1;0,1,1]; %distinguishable_colors(length(sample_uni)+1);
i = 1;
scatter(mapped_xy(:,1),mapped_xy(:,2),3,0.3*[1,1,1],'filled'); hold on;
s = scatter(mapped_xy(side_flag_sorted==side_flag_uni(i),1),mapped_xy(side_flag_sorted==side_flag_uni(i),2),10,colors(i,:),'filled'); hold on;
alpha(s,0.3);
axis tight
axis equal
axis off
legend('ctrl','FC')

axes(ha(2))
i = 2;
scatter(mapped_xy(:,1),mapped_xy(:,2),3,0.3*[1,1,1],'filled'); hold on;
s = scatter(mapped_xy(side_flag_sorted==side_flag_uni(i),1),mapped_xy(side_flag_sorted==side_flag_uni(i),2),10,colors(i,:),'filled'); hold on;
alpha(s,0.3);
axis tight
axis equal
axis off
legend('FC','ctrl')
% plot by tissue
colors = [1,0,0;0,0,1;1,0,0;0,0,1]; %distinguishable_colors(length(sample_uni)+1);
axes(ha(3))
s = scatter(mapped_xy(male_flag_sorted,1),mapped_xy(male_flag_sorted,2),10,colors(1,:),'filled'); hold on;
alpha(s,0.3);
s = scatter(mapped_xy(female_flag_sorted,1),mapped_xy(female_flag_sorted,2),10,colors(2,:),'filled'); hold on;
alpha(s,0.3);
axis tight
axis off
axis equal
legend('Amy','Piri')
if savefig_flag==1
    % eval(['export_fig tsne_AmyPiri_FC_by_sampleannot_','_',date,'.pdf']);
    savefig(gcf,['tsne_all_onNeurons_by_annot_v1_Nneighbors_',num2str(n_neighbors),'_minDist',num2str(min_dist),'_',date,'.fig'])
end
if savefig_pdf==1
    eval(['export_fig tsne_by_annot2_all_neurons_v1_',date,'.pdf']);
end


% % % % % % % % % % % % % % % % % % % % % % % % % % 

figure;
set(gcf,'position',[100,100,1400,770],'color','w')
ax1 = axes('position',[0.1,0.02,0.88,0.84]);
imagesc(datamarkers_cn,[prctile(datamarkers_cn(:),1),prctile(datamarkers_cn(:),99)]);
hold on;
linewid =0.5;
bor_color = 'grey11';%'green1';%
for jj=1:length(cells_bor)
    plot(cells_bor(jj)*[1,1]-0.5,[1,length(gr_tmp_mark)],'-','linewidth',linewid,'color',get_RGB(bor_color))
end
set(gca,'xtick',gr_center,'xticklabel',[1:length(gr_center)],'ytick',[1:length(gr_tmp_mark)],'yticklabel',gr_tmp_mark, 'fontsize', 10)
colormap('summer');
freezeColors(gca);

female_p = zeros(size(side_flag_sorted));
male_p = zeros(size(side_flag_sorted));
left_p = zeros(size(side_flag_sorted));
gad2_frac = zeros(size(side_flag_sorted));
slc17a6_frac = zeros(size(side_flag_sorted));
slc17a7_frac = zeros(size(side_flag_sorted));
for j=1:length(idxuni)
    female_p(idx==idxuni(j)) = -log10(1-p_male(j));
    male_p(idx==idxuni(j)) = -log10(p_male(j));
    left_p(idx==idxuni(j)) = -log10(p_left(j));
    gad2_frac(idx==idxuni(j)) = mean(gad2(idx==idxuni(j))>0);
    slc17a6_frac(idx==idxuni(j)) = mean(slc17a6(idx==idxuni(j))>0);
    slc17a7_frac(idx==idxuni(j)) = mean(slc17a7(idx==idxuni(j))>0);
end

endcolor = 1-[130,2,126]/255;
cmap = 1-repmat([0:63]'/63,1,3).*repmat(endcolor,64,1);
ax2 = axes('position',[0.1,0.86,0.88,0.02]);
imagesc(left_p',[0,4]); hold on;
colormap(cmap);
freezeColors(gca);
set(gca,'xtick',[],'ytick',[1],'yticklabel','Left');

endcolor = 1-[79,160,202]/255;
cmap = 1-repmat([0:63]'/63,1,3).*repmat(endcolor,64,1);
ax3 = axes('position',[0.1,0.88,0.88,0.02]);
imagesc(male_p'); hold on;
colormap(cmap);
freezeColors(gca);
set(gca,'xtick',[],'ytick',[1],'yticklabel','Male');

endcolor = 1-[128,196,28]/255;
cmap = 1-repmat([0:63]'/63,1,3).*repmat(endcolor,64,1);
ax4 = axes('position',[0.1,0.9,0.88,0.02]);
imagesc(female_p'); hold on;
colormap(cmap);
freezeColors(gca);
set(gca,'xtick',[],'ytick',[1],'yticklabel','Female');

endcolor = 1-[254,166,33]/255;
cmap = 1-repmat([0:63]'/63,1,3).*repmat(endcolor,64,1);
ax5 = axes('position',[0.1,0.93,0.88,0.02]);
imagesc(gad2_frac'); hold on;
colormap(cmap);
freezeColors(gca);
set(gca,'xtick',[],'ytick',[1],'yticklabel','Gad2');

ax6 = axes('position',[0.1,0.95,0.88,0.02]);
imagesc(slc17a7_frac'); hold on;
colormap(cmap);
freezeColors(gca);
set(gca,'xtick',[],'ytick',[1],'yticklabel','Slc17a7');

ax7 = axes('position',[0.1,0.97,0.88,0.02]);
imagesc(slc17a6_frac'); hold on;
colormap(cmap);
freezeColors(gca);
set(gca,'xtick',[],'ytick',[1],'yticklabel','Slc17a6');

linkaxes([ax1,ax2,ax3,ax4,ax5,ax6,ax7],'x')
if savefig_flag==1
    savefig(gcf,['markertable2_all_onNeurons_v1_',date,'.fig'])
    % eval(['export_fig markertable_AmyPiri_FC_',date,'.pdf']);
end
if savefig_pdf==1
    eval(['export_fig markertable2_all_onlyNeurons_v1_',date,'.pdf']);
end

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
[~,loc] = ismember({'Egr4','Egr2','Nr4a1','Gadd45g','Fos','Arc','Btg2','Npas4'},geneid_all);

marker = data_orig_all_sorted(loc,:);
marker_percent = zeros(1, length(unique(idx)));
markergene = zeros(1,length(idxuni));
for j=1:length(idxuni)
    %     c1 = sum( idx==idxuni(j));
    %     c2 = sum( idx==idxuni(j) & marker'>0);
    c2 = mean( mean(marker(:,idx==idxuni(j))>0,2) );
    marker_percent(j) = 100*c2;
    %     marker_percent(j) = mean(marker(idx==idxuni(j)));
    markergene(j) = mean(gad2(idx==idxuni(j))>0);
end

markergene_color = [interp1([min(markergene),max(markergene)],[0,c_rgb(1)],markergene'),...
    interp1([min(markergene),max(markergene)],[0,c_rgb(2)],markergene')...
    ,interp1([min(markergene),max(markergene)],[1,c_rgb(3)],markergene')];

figure('position',[100,100,800,770],'color','w'); 
% plot(male_percent,left_percent,'.'); hold on;
scatter(male_percent,left_percent,2*marker_percent,markergene_color,'o','filled'); hold on;
text(male_percent+2,left_percent+2,cellfun(@num2str,m2c([1:91]),'UniformOutput',0),'fontsize',10)
plot([0:100],[0:100]*left_male+[100:-1:0]*left_female,'--k','linewidth',2);
plot(100*mean(male_flag_sorted)*[1,1],[0,100],'--k','linewidth',2)
axis equal
set(gca,'xlim',[0,100],'ylim',[0,100],'xtick',[0:10:100],'ytick',[0:10:100])
xlabel('Amy%');
ylabel('FC%');
box on
if savefig_pdf==1
    eval(['export_fig clusters_FC_vs_Amy_onlyNeurons_v1_',date,'.pdf']);
end


% % % % % % % % % % % % % % % % % % % % % % % % % % 
% set(gcf,'position',[332,64,1251,775]);
% set(gca,'xlim',[1423,1569],'ylim',[85,101])
% 
% eval(['export_fig clusters_18_zoomin_',date,'.pdf']);

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % addressing sample mixing per cluster
sample_uni = {'10-1','18-1','19-1','8-1','23-1','23-3','10-2','18-2','8-2','23-2'};
ContPerSam = zeros(length(sample_uni),length(T_cells_tmp_uni));
for jjj=1:length(T_cells_tmp_uni)
    jjj
    ind = find(T_cells_tmp==T_cells_tmp_uni(jjj));
    ContPerSam(:,jjj) = sum(samples_num(:,ind),2)/length(ind);
end
figure('position',[100,100,1200,300],'color','w');     
ax1 = axes('position',[0.05,0.05,0.65,0.75]);
imagesc(ContPerSam);
set(gca,'ytick',[1:length(sample_uni)],'yticklabel',sample_uni, 'fontsize', 10)
colormap('summer');
freezeColors(gca);

endcolor = 1-[130,2,126]/255;
cmap = 1-repmat([0:63]'/63,1,3).*repmat(endcolor,64,1);
ax2 = axes('position',[0.05,0.81,0.65,0.04]);
imagesc(-log10(p_left),[0,4]); hold on;
colormap(cmap);
freezeColors(gca);
set(gca,'xtick',[],'ytick',[1],'yticklabel','FC');
  
endcolor = 1-[79,160,202]/255;
cmap = 1-repmat([0:63]'/63,1,3).*repmat(endcolor,64,1);
ax3 = axes('position',[0.05,0.86,0.65,0.04]);
imagesc(-log10(p_male)); hold on;
colormap(cmap);
freezeColors(gca);
set(gca,'xtick',[],'ytick',[1],'yticklabel','Amy');

endcolor = 1-[128,196,28]/255;
cmap = 1-repmat([0:63]'/63,1,3).*repmat(endcolor,64,1);
ax3 = axes('position',[0.05,0.91,0.65,0.04]);
imagesc(-log10(p_female)); hold on;
colormap(cmap);
freezeColors(gca);
set(gca,'xtick',[],'ytick',[1],'yticklabel','Piri');


ax4 = axes('position',[0.75,0.05,0.23,0.75]);
d = corr_mat(ContPerSam');
imagesc(d,[-0.2,0.8])
colormap('summer');
set(gca,'xtick',[],'ytick',[]);
hcb = colorbar;

eval(['export_fig sample_contribution_per_cluster_',date,'.pdf']);



% % % % % % % % % % % % % % % % % % % % % % % % %
c1 = 33; c2 = 34;
top_g = 20;
gr1 = find(idx==c1 );
gr2 = find(idx==c2 );
x1 = mean(log2(data_sorted_all(:,gr1)+1),2);
x2 = mean(log2(data_sorted_all(:,gr2)+1),2);
d = x1-x2 ;
[~,xi] = sort(d);
figure('position',[200,200,1000,580],'color','w');
[ha, pos] = tight_subplot(1, 2, [0.05,0.05], [0.1,0.05], [0.05,0.05]);
axes(ha(1))
plot(x1, x2, '.');hold on;
xmax = max(x1);
plot([0,xmax],[0,xmax],'-k'); grid on
plot([0,xmax],[0,xmax]+1,'--k'); grid on
plot([1,xmax],[0,xmax-1],'--k'); grid on
plot(x1(xi(1:top_g)), x2(xi(1:top_g)),'.r'); hold on;
text(x1(xi(1:top_g)), x2(xi(1:top_g)),geneid(xi(1:top_g)),'fontsize',6);
xi = flipud(xi);
plot(x1(xi(1:top_g)), x2(xi(1:top_g)),'.r'); hold on;
text(x1(xi(1:top_g)), x2(xi(1:top_g)),geneid(xi(1:top_g)),'fontsize',6);
title(['cluster ',num2str(c2),' vs ',num2str(c1)])
xlabel(['mean (',num2str(c1),')'])
ylabel(['mean (',num2str(c2),')'])
axis tight

top_g = 20;
gr1 = find(idx==c1 );
gr2 = find(idx==c2 );
x1 = mean(data_sorted_all(:,gr1)>0,2);
x2 = mean(data_sorted_all(:,gr2)>0,2);
d = x1-x2 ;
[~,xi] = sort(d);
axes(ha(2))
plot(x1, x2, '.');hold on;
xmax = max(x1);
plot([0,xmax],[0,xmax],'-k'); grid on
plot([0,xmax],[0,xmax]+0.4,'--k'); grid on
plot([0.4,xmax],[0,xmax-0.4],'--k'); grid on
plot(x1(xi(1:top_g)), x2(xi(1:top_g)),'.r'); hold on;
text(x1(xi(1:top_g)), x2(xi(1:top_g)),geneid(xi(1:top_g)),'fontsize',6);
xi = flipud(xi);
plot(x1(xi(1:top_g)), x2(xi(1:top_g)),'.r'); hold on;
text(x1(xi(1:top_g)), x2(xi(1:top_g)),geneid(xi(1:top_g)),'fontsize',6);
title(['cluster ',num2str(c2),' vs ',num2str(c1)])
xlabel(['mean (',num2str(c1),')'])
ylabel(['mean (',num2str(c2),')'])
axis tight
if savefig_flag==1
    savefig(gcf,['scatter_cluster_',num2str(c1),'_vs_',num2str(c2),'_',date,'.fig'])
end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %


