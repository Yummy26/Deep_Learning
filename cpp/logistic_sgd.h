#include <math.h>
#include "utils.h"
#include "cblas.h"

using namespace std;

class LogisticRegression
{
public:
    double *W;
    double *b;

    LogisticRegression(int n_in, int n_out, int batch_size) 
    {

        W = new double[n_in * n_out];
        b = new double[n_out];

        p_y_given_x = new double[batch_size * n_out];
        y_pred = new double[batch_size];
        I = new double[batch_size];

        this->batch_size = batch_size;
        this->n_in = n_in;
        this->n_out = n_out;

        fill_n(W, n_in * n_out, 0);
        fill_n(b, n_out, 0);
        fill_n(I, batch_size, 1);
    };

    ~LogisticRegression()
    {
        delete[] this->W;
        delete[] this->b;
        delete[] this->p_y_given_x;
        delete[] this->y_pred;
        delete[] this->I;
    };

    double negative_log_likelihood(double *input, double *y)
    {
        get_p_y_given_x(input);   
        double val=0.0;
        for (int b=0; b<batch_size; b++)
        {
            val += log(p_y_given_x[b*n_out+(int)y[b]]);
        }

        return -(val/batch_size);
    };

    double errors(double *input, double *y)
    {
        get_p_y_given_x(input);
        argmax(p_y_given_x);

        double cout = 0;
        for (int i=0; i<batch_size; i++)
            if (y_pred[i] != y[i])
                cout += 1;

        return cout / batch_size;
    };

    void updates(double *input, double *y, double lr)
    {
        get_p_y_given_x(input);

        for (int b=0; b<batch_size; b++)
        {
            for (int n=0; n<n_out; n++)
            {
                if ((int) y[b] == n)
                    p_y_given_x[b*n_out+n] -= 1;
            }
        }
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                n_out, n_in, batch_size, (-1)*lr/batch_size,
                p_y_given_x, n_out, input, n_in,
                1.0, W, n_in);

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans,
                n_out, 1, batch_size, (-1)*lr/batch_size,
                p_y_given_x, n_out, I, batch_size,
                1.0, b, 1);
    };

private:
    int batch_size;
    int n_in;
    int n_out;

    double *p_y_given_x;
    double *y_pred;
    double *I;

    void get_p_y_given_x(double *input)
    {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                batch_size, n_out, n_in, 1.0,
                input, n_in, W, n_in,
                0.0, p_y_given_x, n_out);
                
        cblas_dger(CblasRowMajor,
                batch_size, n_out, 1.0,
                I, 1, b, 1,
                p_y_given_x, n_out);

        softmax(p_y_given_x);
    };

    void softmax(double *p_y_given_x)
    {
        double sum;
        for (int b=0; b<batch_size; b++) 
        {
            sum = 0.0;
            for (int n=0; n<n_out; n++)
            {
                p_y_given_x[b*n_out+n] = exp(p_y_given_x[b*n_out+n]);
                sum += p_y_given_x[b*n_out+n];
            }

            for (int n=0; n<n_out; n++)
            {
                p_y_given_x[b*n_out+n] = p_y_given_x[b*n_out+n]/sum;
            }
        }
    };

    void argmax(double *p_y_given_x)
    {
        int index;
        double val;
        for (int b=0; b<batch_size; b++) 
        {
            index = 1;
            double val = 0.0;
            for (int n=0; n<n_out; n++)
            {
                if (p_y_given_x[b*n_out+n] > val)
                {
                    index = n;
                    val = p_y_given_x[b*n_out+n];
                }
            }
            y_pred[b] = index;
        }
    };
};

