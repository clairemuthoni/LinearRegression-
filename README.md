# Linear Regression 
In this file we are implementing the linear regression algorithm from scratch, using the numpy library. 
So in the beginning we use the numpy library to generate data. 
The data generated, the x is linearly spaces between 0 and 10, and the y we used the formula y = 2x^2^ + 3x + 5 + noise (noise being gaussian noise with mean 0 and standard deviation 5)

## Ordinary Least Mean Squared Method 
This is the first method of calculating the linear regression. So the first step is calculating the mean of the x and y values. These values will help us calculate the gradient and the y-intercept 
we all know the formula of the line is y = mx + c. 
We then apply the formula sum((x - x_mean) * (y - y_mean)) / sum((x - x_mean) ** 2 ) to calculate the slope of the line.
We then apply the formula y_mean - m*x_mean to calculate the y-intercept.
We got the value of m as 23.1140 

and c as -28.9397 

The mean squared error is 244.9987 


## Gradient Descent 
This is the second method of calculating the linear regression. So the process is to begin by setting the m and c to zero and then 
we begin updating the gradient using the formula m = m - learning_rate * dj_dm where dj_dm = (2/n) * sum((y_pred - y) * x)

Then we update the y - intercept c using the formula c = c - learning_rate * dj_dc where dj_dc = (2/n) * sum(y_pred - y)
We compute the cost or MSE as (1/n) * sum((y_pred -y) ** 2) 
