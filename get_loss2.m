function loss = get_loss2(response,lambda_l)

p1 = response(:);
p1(p1>1)=1;
p2 = 1-p1;
logP1 = log(p1);
logP2 = log(p2);
H = p1.*logP1+p2.*logP2;
H = mean(H);
L = max(logP1);
loss = -L-lambda_l*H;
end