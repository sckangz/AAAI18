function [result] =SCMK(A,s,alpha,beta,gamma,mu)
% This is the code for AAAI 2018 paper:Unified Spectral Clustering with Optimal Graph
% wrote by zhao kang,email: zkang@uestc.edu.cn
 addpath('C:\Users\User\Desktop\research\kernelclusteringexp\FOptM-share')

%A is the combined kernel tensor
[m,n,mn]=size(A);%m=n=sample number, mn is kernel number,e.g., 12 in this paper
Y1=zeros(n);
Z=eye(n);
e=1/12*ones(12,1);
c=length(unique(s));
F = randn(n,c);    F= orth(F);
Q = randn(c);    Q= orth(Q);
Y=zeros(n,c);
opts.record = 0;
opts.mxitr  = 1000;%1000
opts.xtol = 1e-5;
opts.gtol = 1e-5;
opts.ftol = 1e-8;
out.tau = 1e-3;

for i=1:200
    i
    K=zeros(n);
    for j=1:12
        K=K+e(j)*A(:,:,j);
    end
    D=Z-Y1/mu;
    W=max(abs(D)-alpha/mu,0).*sign(D);
    W=W-diag(diag(W));
    W(find(W<0))=0;
    E=W+Y1/mu;
    for ij=1:n
        
        [all]=veccomp2(ij,n,F);
        H=mu*eye(n)+2*K;
%         H=(H+H')/2;
        ff=beta/2*all'-2*K(:,ij)-mu*E(:,ij);
        Z(:,ij)=.5*H\(-ff);
        % we use the free package to solve quadratic equation: http://sigpromu.org/quadprog/index.html
%         [Z(:,ij),err,lm] = qpas(H,ff,[],[],ones(1,n),1,zeros(n,1),ones(n,1));
    end
    Z=Z-diag(diag(Z));
    Z(find(Z<0))=0;
    
    Y1=Y1+mu*(W-Z);
    mu=mu*1.1;
    Z= (Z+Z')/2;
    D = diag(sum(Z));
    L = D-Z;
    [F,out]= solveF(F, @fun1,opts,gamma/beta,Y,Q,L);
   
    [a b d]=svd(F'*Y);
    Q=a*d';
    
    Y=zeros(n,c);
    
    for ji=1:n
        P=F*Q;
        [v,j]=max(P(ji,:));
        Y(ji,j)=1;
        l(ji)=j;
    end
    h=zeros(12,1);
    for j=1:12
        h(j)=trace(A(:,:,j)-2*A(:,:,j)*Z+Z'*A(:,:,j)*Z);
    end
    for j=1:12
        e(j)=(h(j)*sum(1./h))^(-2);
        % e(j)=(1/(sum((h).^(gamma/(gamma-1)))))^(1/gamma)*(h(j))^(1/(gamma-1));
    end
    e
    if((i>1)&(norm(Z-W,'fro') < norm(Z,'fro') * 1e-3))
        break
    end
end
[result] = ClusteringMeasure(l,s)
    function [F,G]=fun1(P,alpha,Y,Q,L)
        G=2*L*P-2*alpha*Y*Q';
        F=trace(P'*L*P)+alpha*norm(Y-P*Q,'fro');
    end

    function [F,G]=fun2(Q,P,Y)
        G=-2*P'*Y+2*Q;
        F=norm(Y-P*Q,'fro');
    end
    function [all]=veccomp2(ij,n,F)
        for ji=1:n
            all(ji)=norm(F(ij,:)-F(ji,:));
        end
    end


end
