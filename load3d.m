function [x,y,z,var]=load3d(name)

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

end