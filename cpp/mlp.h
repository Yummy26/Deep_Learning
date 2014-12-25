#include <math.h>
#include <stdlib.h>
#include "logistic_sgd.h"

using namespace std;

class HiddenLayer
{
    public:
        double *W;
        double *b;
        double *p_y_given_x;

        HiddenLayer(int n_in, int n_out, int batch_size)
        {
            this->n_in = n_in;
            this->n_out = n_out;
            this->batch_size = batch_size;

            this->I  = new double[batch_size];
            fill_n(this->I, batch_size, 1);

            this->W = new double[n_in * n_out];
            this->b = new double[n_out];
            this->p_y_given_x = new double[batch_size * n_out];

            double low = -sqrt((double)6/(n_in + n_out));
            double high = sqrt((double)6/(n_in + n_out));

            for (int i=0; i<n_in*n_out; i++) 
            {
                this->W[i] = uniform(low, high);
            }

            fill_n(this->b, n_out, 0);
        };

        void output(double* input)
        {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    batch_size, n_out, n_in, 1.0,
                    input, n_in, W, n_in,
                    0.0, p_y_given_x, n_out);

            cblas_dger(CblasRowMajor,
                    batch_size, n_out, 1.0,
                    I, 1, b, 1,
                    p_y_given_x, n_out);

            for (int i=0; i<batch_size*n_out; i++) p_y_given_x[i] = tanh(p_y_given_x[i]);
        };

        ~HiddenLayer()
        {
            delete[] I;
            delete[] W;
            delete[] b;
            delete[] p_y_given_x;
        };

    private:
        int n_in;
        int n_out;
        int batch_size;

        double* I;

        double uniform(double low, double high)
        {
            return rand() / (RAND_MAX + 1.0) * (high - low) + low;
        };

        double tanh(double x)
        {
            return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
        };
};

class MLP
{
    public:
        HiddenLayer* hiddenLayer; 
        LogisticRegression* logisticRegression;

        MLP(int n_in, int n_hidden, int n_out, int batch_size)
        {
            this->n_in = n_in;
            this->n_hidden = n_hidden;
            this->n_out = n_out;
            this->batch_size = batch_size;
             

            hiddenLayer = new HiddenLayer(n_in, n_hidden, batch_size);
            logisticRegression = new LogisticRegression(n_hidden, n_out, batch_size);

            this->I = new double[batch_size];
            this->deltak = new double[batch_size * n_out];
            this->deltaj = new double[batch_size * n_hidden];
        };

        void update(double *input, double *y, double lr, double L1_reg, double L2_reg)
        {
            hiddenLayer->output(input);
            logisticRegression->output(hiddenLayer->p_y_given_x);

            for (int b=0; b<batch_size; b++)
            {
                for (int n=0; n<n_out; n++)
                {
                    deltak[b*n_out+n] = logisticRegression->p_y_given_x[b*n_out+n];
                    if ((int) y[b] == n)
                        deltak[b*n_out+n] = deltak[b*n_out+n] - 1;
                }
            }

               cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
               batch_size, n_hidden, n_out, 1,
               deltak, n_out, logisticRegression->W, n_hidden,
               0, deltaj, n_hidden);

               for (int i=0; i<batch_size*n_hidden; i++)
                   deltaj[i] = deltaj[i] * (1 - pow(hiddenLayer->p_y_given_x[i], 2));

               cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
               n_out, n_hidden, batch_size, (-1)*lr/batch_size,
               deltak, n_out, hiddenLayer->p_y_given_x, n_hidden,
               1.0-2*L2_reg*lr/batch_size, logisticRegression->W, n_hidden);

               cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans,
               n_out, 1, batch_size, (-1)*lr/batch_size,
               deltak, n_out, I, batch_size,
               1.0, logisticRegression->b, 1);

               cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
               n_hidden, n_in, batch_size, (-1)*lr/batch_size,
               deltaj, n_hidden, input, n_in,
               1.0-2*L2_reg*lr/batch_size, hiddenLayer->W, n_in);

               cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans,
               n_hidden, 1, batch_size, (-1)*lr/batch_size,
               deltaj, n_hidden, I, batch_size,
               1.0, hiddenLayer->b, 1);

        };

        double negative_log_likelihood(double *input, double *y, double L1_reg, double L2_reg)
        {
            hiddenLayer->output(input);
            return logisticRegression->negative_log_likelihood(hiddenLayer->p_y_given_x, y) + L1_reg * L1() + L2_reg * L2_sqr();
        };

        double errors(double *input, double *y)
        {
            hiddenLayer->output(input);
            return logisticRegression->errors(hiddenLayer->p_y_given_x, y);
        };

        ~MLP()
        {
            delete hiddenLayer;
            delete logisticRegression;

            delete[] I;
            delete[] deltak;
            delete[] deltaj;
        };

    private:
        int n_in;
        int n_hidden;
        int n_out;
        int batch_size;

        double* I;
        double* deltak;
        double* deltaj;

        double L1()
        {
            double sum = 0.0;
            for (int i=0; i<n_in*n_hidden; i++)
            {
                sum += fabs(hiddenLayer->W[i]);
            }

            for (int i=0; i<n_hidden*n_out; i++)
            {
                sum += fabs(logisticRegression->W[i]);
            }

            return sum;
        };

        double L2_sqr()
        {
            double sum = 0.0;
            for (int i=0; i<n_in*n_hidden; i++)
            {
                sum += (hiddenLayer->W[i]) * (hiddenLayer->W[i]);
            }

            for (int i=0; i<n_hidden*n_out; i++)
            {
                sum += (logisticRegression->W[i]) * (logisticRegression->W[i]);
            }

            return sum;
        };

};
