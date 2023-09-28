import torch
from einops import rearrange

from model.snn_layers import first_order_low_pass_layer, neuron_layer


class SNN_Model(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.batchsize = cfg.batchsize

        self.axon1 = first_order_low_pass_layer((cfg.dim_1,), cfg.length, self.batchsize, cfg.tau_m,
                                                cfg.train_coefficients)
        self.snn1 = neuron_layer(cfg.dim_1, cfg.dim_2, cfg.length, self.batchsize, cfg.tau_m, cfg.train_bias,
                                 cfg.membrane_filter)

        self.axon2 = first_order_low_pass_layer((cfg.dim_2,), cfg.length, self.batchsize, cfg.tau_m,
                                                cfg.train_coefficients)
        self.snn2 = neuron_layer(cfg.dim_2, 300, cfg.length, self.batchsize, cfg.tau_m, cfg.train_bias,
                                 cfg.membrane_filter)

        self.axon3 = first_order_low_pass_layer((300,), cfg.length, self.batchsize, cfg.tau_m, cfg.train_coefficients)
        self.snn3 = neuron_layer(300, 100, cfg.length, self.batchsize, cfg.tau_m, cfg.train_bias, cfg.membrane_filter)

        self.dropout1 = torch.nn.Dropout(p=0.1, inplace=False)
        self.dropout2 = torch.nn.Dropout(p=0.1, inplace=False)
        self.linear = torch.nn.Linear(100 * cfg.in_channels, cfg.n_classes)

    def forward(self, inputs):
        """
        :param inputs: [batch, input_size, t]
        :return:
        """
        inputs = rearrange(inputs, 'b c h -> b h c')  # 16,62,900->16,900,62
        axon1_states = self.axon1.create_init_states()
        snn1_states = self.snn1.create_init_states()

        axon2_states = self.axon2.create_init_states()
        snn2_states = self.snn2.create_init_states()

        axon3_states = self.axon3.create_init_states()
        snn3_states = self.snn3.create_init_states()

        axon1_out, axon1_states = self.axon1(inputs, axon1_states)  # 16,900,62   16,900
        spike_l1, snn1_states = self.snn1(axon1_out, snn1_states)  # 16,500,62  16,500  16,500
        drop_1 = self.dropout1(spike_l1)  # 16,500,62

        axon2_out, axon2_states = self.axon2(drop_1, axon2_states)  # 16,500,62  16,500
        spike_l2, snn2_states = self.snn2(axon2_out, snn2_states)  # 16,300,62  16,300  16,300
        drop_2 = self.dropout2(spike_l2)  # 16,300,62

        axon3_out, axon3_states = self.axon3(drop_2, axon3_states)  # 16,300,62  16,300
        spike_l3, snn3_states = self.snn3(axon3_out, snn3_states)  # 16,100,62  16,100  16,100
        spike_l3 = spike_l3.reshape(self.batchsize, -1)  # 16,6200
        spike_l3 = self.linear(spike_l3)  # 16,3
        return spike_l3
