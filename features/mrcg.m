function [ cg ] = mrcg( gf1 , gf2 )
%根据GF特征计算MRCG特征
%
[m,n]=size(gf1);

% coef=1/15;
% gf1=gf1.^coef;
% gf2=gf2.^coef;
% gf1=log(gf1);
% gf2=log(gf2);

width=11;
halfwidth=width/2-0.5;
gf=zeros(m,n);
for k=1:m
    for j=1:n
        left=max(1,j-halfwidth);
        right=min(n,j+halfwidth);
        up=max(1,k-halfwidth);
        down=min(m,k+halfwidth);
        inside=gf1(up:down,left:right);
        gf(k,j)=sum(sum(inside))/width/width;
    end
end
gf3=gf;

width=23;
halfwidth=width/2-0.5;
gf=zeros(m,n);
for k=1:m
    for j=1:n
        left=max(1,j-halfwidth);
        right=min(n,j+halfwidth);
        up=max(1,k-halfwidth);
        down=min(m,k+halfwidth);
        inside=gf1(up:down,left:right);
        gf(k,j)=sum(sum(inside))/width/width;
    end
end
gf4=gf;

cg=[gf1;gf2;gf3;gf4];
% cg=[gf1;gf3];%;gf4];
end

