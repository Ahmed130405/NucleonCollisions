clear all;
% contour plot of <e3>/<e2> for different (d,w) pairs (0 to 1% centrality data only)
for i = 1:6
    for j = 1:5
        % can find data via link in file largeData.txt
        filenameString = ['Pb_Pb_d' num2str(0.6+i*0.1) '_w' num2str(0.2+j*0.2) '_bmax1.7427.txt']; % b=1.7427 estimate for 1% percentile impact paramater
        data = readtable(filenameString);
        e2 = table2array(data(:,5:5));
        e2rms = rms(e2);
        e3 = table2array(data(:,6:6));
        e3rms = rms(e3);
        z(i,j) = e3rms/e2rms;
    end
end
d = 0.7:0.1:1.2;
w = 0.4:0.2:1.2;
[D,W] = meshgrid(d,w);
% use contour plot to find estimate for w
figure(1), clf
contourf(transpose(D),transpose(W),z)
c = colorbar;
c.Label.String = 'e3/e2';
xlabel('d');
ylabel('w');

% plot <e3>/<e2> vs. centrality (0 to 10% in 1% bins)
filenameString = ['Pb_Pb_d0.8_w0.4_5TeV.txt']; % w=0.4 estimate
data = readtable(filenameString);
b = table2array(data(:,2:2));
for i = 1:10
    bprc(i) = prctile(b,i);
end
bprc = transpose(bprc);
filenameString = ['Pb_Pb_d0.8_w0.4_5TeV_bmax' num2str(bprc(1)) '.txt'];
data = readtable(filenameString);
e2 = table2array(data(:,5:5));
e2rms = rms(e2);
e3 = table2array(data(:,6:6));
e3rms = rms(e3);
z2(1) = e3rms/e2rms;
for i = 2:10
    filenameString = ['Pb_Pb_d0.8_w0.4_5TeV_bmin' num2str(bprc(i-1)) '_bmax' num2str(bprc(i)) '.txt'];
    data = readtable(filenameString);
    e2 = table2array(data(:,5:5));
    e2rms = rms(e2);
    e3 = table2array(data(:,6:6));
    e3rms = rms(e3);
    z2(i) = e3rms/e2rms;
end
z2 = transpose(z2);
T = table(bprc,z2);
T.Properties.VariableNames = {'b' 'e3(2)/e2(2)'};
writetable(T,'e3_e2_table_1');
figure(2), clf
scatter(bprc,z2)
xlabel('b');
ylabel('e3(2)/e2(2)');

% find estimate for optimum d using BG fit
for j = 1:3
    for i = 1:6
        filenameString = ['Pb_Pb_d' num2str(0.6+i*0.1) '_w0.4.txt'];
        data = readtable(filenameString);
        b = table2array(data(:,2:2));
        if j == 1
            I = b < prctile(b,5*j);
        else
            I = b > prctile(b,5*(j-1)) & b < prctile(b,5*j);
        end
        e2 = table2array(data(:,5:5)); e2 = e2(I); e2 = e2/mean(e2);
        figure(6*(j-1)+i+2), clf
        h = histogram(e2,50,'Normalization','pdf');
        xlabel('e2norm');
        titleString = ['trento cent' num2str(5*(j-1)) '-' num2str(5*j) ' d' num2str(0.6+i*0.1)];
        title(titleString);
        Y{j,i} = h.Values;
        binEdges = h.BinEdges;
        X{j,i} = binEdges(1:end-1) + h.BinWidth/2;
    end
end
for j = 1:3
    for i = 1:6
        x = transpose(X{j,i});
        y = transpose(Y{j,i});
        a0 = [0 1];
        fitfun = fittype( @(a,b,x) BG(x,a,b) );
        [fitted_curve,gof] = fit(x,y,fitfun,'StartPoint',a0);
        coeffvals = coeffvalues(fitted_curve);
        trentocoeffs{j,i} = coeffvals;
    end
end
for j = 1:3
    filenameString = ['atlas_cent' num2str(5*(j-1)) '-' num2str(5*j) '.csv'];
    data = readtable(filenameString);
    v2 = table2array(data(:,1:1)); v2 = v2/mean(v2);
    probv2 = table2array(data(:,2:2)); probv2 = probv2/trapz(v2,probv2);
    U{j} = v2;
    V{j} = probv2;
end
for j = 1:3
    x = U{j};
    y = V{j};
    a0 = [0 1];
    fitfun = fittype( @(a,b,x) BG(x,a,b) );
    [fitted_curve,gof] = fit(x,y,fitfun,'StartPoint',a0);
    coeffvals = coeffvalues(fitted_curve);
    atlascoeffs{j} = coeffvals;
end
for j = 1:3
    d = 0.7:0.1:1.2;
    for i = 1:6
        trentosigmavals(i) = trentocoeffs{j,i}(2);
    end
    atlassigmaval = atlascoeffs{j}(2);
    figure(20+j), clf
    scatter(d, trentosigmavals)
    hold on;
    p = polyfit(d,trentosigmavals,1); % linear fit
    plot(d,polyval(p,d));
    xlabel('d');
    ylabel('sigma');
    titleString = ['cent' num2str(5*(j-1)) '-' num2str(5*j)];
    title(titleString);
    atlasdvals(j) = (atlassigmaval - p(2))/p(1); % extrapolate + solve for LHS in lin eq
end
atlasdvals = transpose(atlasdvals);
cent = 5:5:15; cent = transpose(cent);
T = table(cent,atlasdvals);
T.Properties.VariableNames = {'cent' 'atlasdvals'};
writetable(T,'atlasdvals_1');
for i = 1:23
    figurenameString = ['-f' num2str(i)];
    print(figurenameString,'-dpng');
end

% filter 0 to 1% centrality data using entropy (nch) not impact parameter (b)
filenameString = ['Pb_Pb_d0.8_w0.4_5TeV.txt'];
data = readtable(filenameString);
nch = table2array(data(:,4:4));
centrality = 0.5:1:9.5;
centrality = transpose(centrality);0
for i = 1:10
    nchprc(i) = prctile(nch,100-i);
end
for i = 1:10
    if i == 1
        I = nch > prctile(nch,100-i);
    else
        I = nch > prctile(nch,100-i) & nch < prctile(nch,100-(i-1));
    end
    e2 = table2array(data(:,5:5)); e2 = e2(I);
    e3 = table2array(data(:,6:6)); e3 = e3(I);
    e2rms(i) = rms(e2);
    e3rms(i) = rms(e3);
end
T = table(centrality,transpose(e2rms),transpose(e3rms));
T.Properties.VariableNames = {'centrality' 'e2(2)' 'e3(2)'};
writetable(T,'e3_e2_table_2');
U = table(centrality,transpose(nchprc));
U.Properties.VariableNames = {'centrality' 'nch'};
writetable(U,'nchprc_1');
