#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

class Dataset
{
    public:
        int trn_smp;
        int vld_smp;
        int tst_smp;
        int inp_dim;
        int lbl_dim;

        double *trn_dat;
        double *trn_lbl;
        double *vld_dat;
        double *vld_lbl;
        double *tst_dat;
        double *tst_lbl;

        Dataset(int train_samples, int valid_samples, int test_samples, int input_dimension, int label_dimension)
        {
            trn_smp = train_samples;
            vld_smp = valid_samples;
            tst_smp = test_samples;
            inp_dim = input_dimension;
            lbl_dim = label_dimension;

            trn_dat = new double[trn_smp * inp_dim];
            trn_lbl = new double[trn_smp * lbl_dim];
            vld_dat = new double[vld_smp * inp_dim];
            vld_lbl = new double[vld_smp * lbl_dim];
            tst_dat = new double[tst_smp * inp_dim];
            tst_lbl = new double[tst_smp * lbl_dim];
        };

        ~Dataset()
        {
            delete[] trn_dat;
            delete[] trn_lbl;
            delete[] vld_dat;
            delete[] vld_lbl;
            delete[] tst_dat;
            delete[] tst_lbl;
        };

        void load_data_from_binary_file(double *des, string fname, int num, int offset)
        {
            fstream fin;
            fin.open(fname.c_str(), fstream::in | fstream::binary);
            if (fin.is_open())
            {
                fin.seekg(offset, ios_base::beg);
                char* buffer= new char[num];
                fin.read(buffer, sizeof(char) * num);

                unsigned char tmp;
                for(int i=0; i<num; i++)
                {
                    tmp = buffer[i];
                    if (fname.find("images") != string::npos)
                        des[i] = (double) tmp/255.0;
                    if (fname.find("labels") != string::npos) 
                        des[i] = (double) tmp;
                }
                delete[] buffer;
            }
            else
            {
                cerr << "Can't load file -- " << fname << endl;
            }
            fin.close();
        };
};
