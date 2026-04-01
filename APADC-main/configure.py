def get_default_config(data_name):
    if data_name in ['MNIST_USPS']:
        return dict(
            Autoencoder=dict(
                arch1=[784, 1024, 1024, 1024, 32],
                arch2=[784, 1024, 1024, 1024, 32],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=0.3,
                seed=0,
                batch_size=256,
                epoch=400,
                lr=1.0e-4,
                lambda1=100,
                lambda2=0.1,
                kernel_mul=2,
                kernel_num=6,
            ),
        )
    elif data_name in ['Caltech101-20']:
        return dict(
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],
                arch2=[512, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=6,
                missing_rate=0.3,
                batch_size=1024,
                epoch=1000,
                lr=1.0e-4,
                lambda1=10,
                lambda2=0.1,
                kernel_mul=2,
                kernel_num=4,
            ),
        )
    elif data_name in ['RGB-D']:
        return dict(
            Autoencoder=dict(
                arch1=[300, 1024, 1024, 1024, 64],
                arch2=[2048, 1024, 1024, 1024, 64],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=0.3,
                seed=8,
                batch_size=256,
                epoch=2000,
                lr=1.0e-4,
                lambda1=10,
                lambda2=0.1,
                kernel_mul=0.01,
                kernel_num=3,
            ),
        )
    elif data_name in ['Scene-15']:
        return dict(
            Autoencoder=dict(
                arch1=[59, 1024, 1024, 1024, 16],
                arch2=[20, 1024, 1024, 1024, 16],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=0.3,
                seed=16,
                batch_size=512,
                epoch=500,
                lr=1.0e-4,
                lambda1=10,
                lambda2=0.1,
                kernel_mul=0.01,
                kernel_num=3,
            ),
        )
    elif data_name in ['NoisyMNIST']:
        return dict(
            Autoencoder=dict(
                arch1=[784, 1024, 1024, 1024, 32],
                arch2=[784, 1024, 1024, 1024, 32],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=0.3,
                seed=2,
                batch_size=512,
                epoch=50,
                lr=1.0e-4,
                lambda1=10.0,
                lambda2=0.1,
                kernel_mul=0.01,
                kernel_num=3,
            ),
        )
    elif data_name in ['ORL']:
        return dict(
            Autoencoder=dict(
                arch1=[512, 1024, 1024, 1024, 32],
                arch2=[59, 1024, 1024, 1024, 32],
                arch3=[864, 1024, 1024, 1024, 32],
                arch4=[254, 1024, 1024, 1024, 32],
                activations1='relu',
                activations2='relu',
                activations3='relu',
                activations4='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=0.1,
                seed=0,
                batch_size=128,
                epoch=400,
                lr=1.0e-4,
                lambda1=100,
                lambda2=0.1,
                kernel_mul=2,
                kernel_num=6,
            ),
        )
    elif data_name in ['COIL20']:
        return dict(
            Autoencoder=dict(
                arch1=[1024, 1024, 1024, 1024, 32],
                arch2=[1024, 1024, 1024, 1024, 32],
                arch3=[324, 1024, 1024, 1024, 32],
                activations1='relu',
                activations2='relu',
                activations3='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=0.1,
                seed=0,
                batch_size=128,
                epoch=400,
                lr=1.0e-4,
                lambda1=100,
                lambda2=100,
                kernel_mul=2,
                kernel_num=4,
            ),
        )
    elif data_name in ['handwritten-5view']:
        return dict(
            Autoencoder=dict(
                arch1=[240, 1024, 1024, 1024, 32],
                arch2=[76, 1024, 1024, 1024, 32],
                arch3=[216, 1024, 1024, 1024, 32],
                arch4=[47, 1024, 1024, 1024, 32],
                arch5=[64, 1024, 1024, 1024, 32],
                activations1='relu',
                activations2='relu',
                activations3='relu',
                activations4='relu',
                activations5='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=0.1,
                seed=0,
                batch_size=256,
                epoch=500,
                lr=1.0e-4,
                lambda1=100,
                lambda2=100,
                kernel_mul=2,
                kernel_num=4,
            ),
        )
    elif data_name in ['100leaves_3v']:
        return dict(
            Autoencoder=dict(
                arch1=[64, 1024, 1024, 1024, 32],
                arch2=[64, 1024, 1024, 1024, 32],
                arch3=[64, 1024, 1024, 1024, 32],
                activations1='relu',
                activations2='relu',
                activations3='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=0.1,
                seed=0,
                batch_size=128,
                epoch=400,
                lr=1.0e-4,
                lambda1=100,
                lambda2=100,
                kernel_mul=2,
                kernel_num=6,
            ),
        )
    elif data_name in ['MSRC_v1']:
        return dict(
            Autoencoder=dict(
                arch1=[24, 1024, 1024, 1024, 32],
                arch2=[576, 1024, 1024, 1024, 32],
                arch3=[512, 1024, 1024, 1024, 32],
                arch4=[256, 1024, 1024, 1024, 32],
                arch5=[254, 1024, 1024, 1024, 32],
                activations1='relu',
                activations2='relu',
                activations3='relu',
                activations4='relu',
                activations5='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=0.1,
                seed=0,
                batch_size=128,
                epoch=500,
                lr=1.0e-4,
                lambda1=10,
                lambda2=10,
                kernel_mul=0.001,
                kernel_num=6,
            ),
        )
    else:
        raise Exception('Undefined data_name')
