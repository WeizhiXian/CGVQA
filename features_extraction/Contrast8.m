function [contrast, Icontrast] = Contrast8(I) 

[m,n] = size(I);
g = padarray(I,[1 1],'symmetric','both');

[r,c] = size(g);
g = double(g);
for i=2:r-1
    for j=2:c-1
        Icontrast(i-1,j-1) = (g(i,j-1)-g(i,j))^2+(g(i-1,j)-g(i,j))^2+(g(i,j+1)-g(i,j))^2+(g(i+1,j)-g(i,j))^2+...
            (g(i-1,j-1)-g(i,j))^2+(g(i-1,j+1)-g(i,j))^2+(g(i+1,j-1)-g(i,j))^2+(g(i+1,j+1)-g(i,j))^2;
        Icontrast(i-1,j-1) = sqrt( Icontrast(i-1,j-1)/8);
    end
end
contrast(1) = sum(sum(Icontrast))/(m*n);
contrast(2) = max(max(Icontrast));

Icontrast = Icontrast/contrast(2);
end