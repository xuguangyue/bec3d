function [x,y,z,var]=load3d(name,flag)

% data = dlmread('real3d-tmpwf_20.txt');
% flag = 1;

data=dlmread([name,'.txt']);
x=unique(data(:,1));
Nx=length(x);

y=unique(data(:,2));
Ny=length(y);

z=unique(data(:,3));
Nz=length(z);

[~,ss]=size(data);
if ss==5
    var=reshape(data(:,ss-1)+1i*data(:,ss),Nz,Ny,Nx);
else
    if ss==4 
        var=reshape(data(:,ss),Nz,Ny,Nx);
    end
end

if flag
    x=[x;-x(Nx-1:-1:1)];
    y=[y;-y(Ny-1:-1:1)];
    z=[z;-z(Nz-1:-1:1)];
    var = [var;var(Nz-1:-1:1,:,:)];
    var = [var,var(:,Ny-1:-1:1,:)];
    var = cat(3,var,var(:,:,Nx-1:-1:1));
end
    

end