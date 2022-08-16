function y = BG(x,a,b)
y = (x./b^2).*besseli(0,a*(x/b^2)).*exp(-((x.^2+a^2)/(2*b^2)));
% width is given by b
end
