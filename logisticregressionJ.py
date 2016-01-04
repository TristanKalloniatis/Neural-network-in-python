def J(theta,X,y,lambda):
  #J(theta, X, y, lambda) computes the cost of using theta as the parameter for regularized logistic regression
  m = length(y) % number of training examples
  temp=sigmoid(np.dot(X,theta))
  J=sum(-y*log(temp)+(y-1)*log(1-temp))
  t2=theta*theta
  t2(0)=0
  J=J+lambda*sum(t2)/2
  return J/m

def gradJ(theta,X,y,lambda):
  # and the gradient of the cost w.r.t. to the parameters. 
  m=length(y)
  temp=sigmoid(np.dot(X,theta))-y
  grad=X.transpose()*temp
  t2=theta
  t2(0)=0
  grad=grad+lambda*t2
  return grad/m
