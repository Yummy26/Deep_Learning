#include "mlp.h"
#include <iostream>
#include <time.h>
#include <string>

using namespace std;

void test_mlp(double learning_rate=0.01, double L1_reg = 0.00, double L2_reg = 0.0001, int n_epochs=1000,
        string dataset="mnist.pk.gz", int n_in = 28*28, int n_hidden = 500, int n_out = 10, int batch_size = 20)
{
    srand(226);

    Dataset datasets(50000, 10000, 10000, n_in, 1);
    datasets.load_data_from_binary_file(datasets.trn_dat, "../data/train-images-idx3-ubyte", 50000*n_in, 16);
    datasets.load_data_from_binary_file(datasets.trn_lbl, "../data/train-labels-idx1-ubyte", 50000, 8);
    datasets.load_data_from_binary_file(datasets.vld_dat, "../data/train-images-idx3-ubyte", 10000*n_in, 50000*n_in+16);
    datasets.load_data_from_binary_file(datasets.vld_lbl, "../data/train-labels-idx1-ubyte", 10000, 50000+8);
    datasets.load_data_from_binary_file(datasets.tst_dat, "../data/t10k-images-idx3-ubyte", 10000*n_in, 16);
    datasets.load_data_from_binary_file(datasets.tst_lbl, "../data/t10k-labels-idx1-ubyte", 10000, 8);

    int n_train_batches = datasets.trn_smp / batch_size;
    int n_valid_batches = datasets.vld_smp / batch_size;
    int n_test_batches = datasets.vld_smp / batch_size;

    MLP classifier(n_in, n_hidden, n_out, batch_size);

    int patience = 10000;
    int patience_increase = 2;
    double improvement_threshold = 0.995;

    int validation_frequency = (n_train_batches < patience/2) ? n_train_batches : patience/2;

    double minibatch_avg_cost;
    double validation_losses = 0.0, this_validation_loss = 0.0, best_validation_loss = 100000;
    double test_losses = 0.0, test_score = 0.0;
    time_t start_time = time(NULL);

    bool done_looping = false;
    int epoch = 0, iter;
    double *trn_x_input, *trn_y_input, *vld_x_input, 
           *vld_y_input, *tst_x_input, *tst_y_input;

    while ((epoch < n_epochs) && (!done_looping))
    {
        epoch = epoch + 1;
        for (int minibatch_index=0; minibatch_index<n_train_batches; minibatch_index++)
        {
            //train part
            trn_x_input = datasets.trn_dat + minibatch_index * batch_size * n_in;
            trn_y_input = datasets.trn_lbl + minibatch_index * batch_size; 

            classifier.update(trn_x_input, trn_y_input, learning_rate, L1_reg, L2_reg);
            iter = (epoch - 1) * n_train_batches + minibatch_index;

            if ((iter + 1) % validation_frequency == 0)
            {
                // validation part
                validation_losses = 0.0;
                for (int vld_batch_index=0; vld_batch_index<n_valid_batches; vld_batch_index++)
                {
                    vld_x_input = datasets.vld_dat + vld_batch_index * batch_size * n_in;
                    vld_y_input = datasets.vld_lbl + vld_batch_index * batch_size;

                    validation_losses += classifier.errors(vld_x_input, vld_y_input);
                }
                this_validation_loss = validation_losses/n_valid_batches;
                cout << "epoch " << epoch << ", minibatch " << minibatch_index+1 << "/" << n_train_batches 
                    << ", validation error " << this_validation_loss * 100 << "%" << endl;


                if (this_validation_loss < best_validation_loss)
                {
                    if (this_validation_loss < best_validation_loss * improvement_threshold)
                        patience = (patience > iter * patience_increase) ? patience : iter * patience_increase;

                    best_validation_loss = this_validation_loss;

                    // test part
                    test_losses = 0.0;
                    for (int tst_batch_index=0; tst_batch_index<n_test_batches; tst_batch_index++)
                    {
                        tst_x_input = datasets.tst_dat + tst_batch_index * batch_size * n_in;
                        tst_y_input = datasets.tst_lbl + tst_batch_index * batch_size;

                        test_losses += classifier.errors(tst_x_input, tst_y_input);
                    }
                    test_score = test_losses/n_test_batches;

                    cout << "\tepoch " << epoch << ", minibatch " << minibatch_index+1 << "/" << n_train_batches 
                        << ", test error of best model " << test_score * 100 << "%" << endl;
                }
            }

            if (patience <= iter)
            {
                done_looping = true;
                break;
            }
        }
    }
    time_t end_time = time(NULL);

    cout << "Optimization complete with best validation score of " << best_validation_loss * 100 
        << "%, with test performance " << test_score * 100 << "%" << endl;
    cout << "The code run for " << epoch << "epochs, with " << 1.0 * epoch /(end_time - start_time) << " epochs/sec "<< endl;
}


int main()
{
    test_mlp();

    return 0;
}

