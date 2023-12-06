
import traceback
from datetime import datetime
from argparse import ArgumentParser

from torch_sparse import SparseTensor
from tqdm import tqdm

from model import OGNNMDA
from calc_metrics import get_metrics
from dataSet import *
from util import *


def train_model(config_path: Path, result_path: Path, logger: Logger):
    logger.info('start training'.center(80, '='))
    start_time = datetime.now().strftime('%Y%m%d%H%M%S')
    logger.info(f'time: {start_time}')
    config = get_config(config_path)
    result_path = result_path / f'{start_time}-{config["desc"]}'
    logger.debug(f'create result dir: {result_path}')
    result_path.mkdir(parents=True)
    save_config(result_path / config_path.name, config)
    model_params = config['model']
    logger.info(f'model parameters: {model_params}')
    train_params = config['train']
    logger.info(f'train parameters: {train_params}')
    dataset_params = config['dataset']
    logger.info(f'dataset: {dataset_params}')
    dataset_params['data_dir'] = Path(dataset_params['data_dir'])
    device = torch.device('cuda' if train_params['use_gpu'] and torch.cuda.is_available() else 'cpu')
    logger.info(f'use: {device}')
    logger.debug('load adj matrix')
    adj = get_drug_microbe_adj_tensor(dataset_params['data_dir'], dataset_params['name'], device)
    pos_weight = ((adj.shape[0]*adj.shape[1] - adj.sum()) / adj.sum())
    score = np.zeros(adj.shape[0]*adj.shape[1])
    label = np.zeros_like(score)
    current = 0
    metrics = []
    cv_index = 0
    logger.info('Splitting dataset ...')
    for train_adj, test_index in k_fold_split(train_params['k_folds'],adj):
        cv_index += 1
        logger.info(f'------this is {cv_index}th cross validation------')
        logger.debug('construct training data')
        data = get_PyG_Data(dataset_params['data_dir'], dataset_params['name'], train_adj, device=device)
        logger.info(data)
        model_params['in_channel'] = data.num_node_features
        assert data.num_drugs == dataset_params['drugs']
        assert data.num_microbes == dataset_params['microbes']
        model_params['drugs'] = dataset_params['drugs']
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug('construct model')
        model = OGNNMDA(params=model_params).to(device)
        logger.debug(f'model name:{model.__class__.__name__}')
        train_adj = train_adj.reshape(-1)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if train_params['weight_decay2'] == "None":
            optimizer = torch.optim.Adam(model.parameters(), lr=train_params['learning_rate'],
                                         weight_decay=train_params['weight_decay'])
        else:
            optimizer = torch.optim.Adam([dict(params=model.params_conv, weight_decay=train_params['weight_decay']),
                                          dict(params=model.params_others, weight_decay=train_params['weight_decay2'])],
                                         lr=train_params['learning_rate'])
        logger.info(f'cv-{cv_index} training start')
        time.sleep(2)
        # loss_lst = [0] * train_params['epochs']
        val_labels = adj.reshape(-1)[test_index].to(dtype=torch.float32, device=device)
        edge_index = SparseTensor.from_dense(data.h_net)
        with tqdm(total=train_params['epochs'], unit='epochs') as pbar:
            for epoch in range(train_params['epochs']):
                model.train()
                optimizer.zero_grad()
                outputs = model(data.x, edge_index)
                train_loss = criterion(outputs, train_adj)
                train_loss.backward()
                optimizer.step()
                # scheduler.step()
                val_loss_str = ''
                pbar.set_description(f't:{train_loss:.5f}{val_loss_str}')
                pbar.update(1)
        logger.info('training done')
        logger.info(f'start cv-{cv_index} validation ...')
        model.eval()
        with torch.no_grad():
            outputs = model(data.x, edge_index)[test_index]
            val_loss = criterion(outputs, val_labels)
            logger.info(f'Val loss: {val_loss:.4f}')
            outputs = torch.sigmoid(outputs)
        logger.info(f'test size:{test_index.shape}')
        metric = get_metrics(*transform(val_labels.cpu().numpy(), outputs.cpu().numpy()))
        logger.info(f'aupr: {metric[0]:.4f} auc: {metric[1]:.4f} f1: {metric[2]:.4f} '
                    f'accuracy: {metric[3]:.4f} recall:{metric[4]:.4f} '
                    f'specificity:{metric[5]:.4f} precision: {metric[6]:.4f}')
        metrics.append(metric)
        label[current:current + val_labels.shape[0]] = val_labels.cpu().numpy()
        score[current:current + outputs.shape[0]] = outputs.cpu().numpy()
        current += val_labels.shape[0]
        logger.info(f'cv-{cv_index} validation done')
    logger.info('statics metrics')
    total = np.array(metrics)
    # label,score = transform((label,score))
    logger.info('[avg] {}'.format(str(total.mean(axis=0)).replace('\n', '')))
    logger.info('[max] {}'.format(str(total.max(axis=0, initial=0)).replace('\n', '')))
    logger.info('[min] {}'.format(str(total.min(axis=0, initial=1)).replace('\n', '')))
    logger.info('save the cross validation experiment results ...')
    cv_metric_res_file = result_path / f'{train_params["k_folds"]}-fold-cv-metrics.txt'
    logger.debug(f'save {train_params["k_folds"]} metric vals: {cv_metric_res_file}')
    np.savetxt(cv_metric_res_file,total)
    logger.debug('save final pred score')
    pred_file = result_path / 'pred.txt'
    label_file = result_path / 'label.txt'
    np.savetxt(label_file,label)
    np.savetxt(str(pred_file),score)
    logger.debug('calc finale score metric')
    metric = get_metrics(*transform(label,score))
    logger.info(f'final pred metric: {metric}')
    pred_metric_file = result_path / f'{train_params["k_folds"]}-fold-final-metric.txt'
    logger.debug(f'save final pred metric: {pred_metric_file}')
    np.savetxt(str(pred_metric_file),metric)
    logger.debug(f'save roc pr curve to: {result_path}')
    logger.info(f'ROC & PR Curve has been saved to {result_path}: *.png')
    logger.warning('train method end')
    return result_path.stem,metric


if __name__ == '__main__':
    args_parser = ArgumentParser(description='OGNNMDA')
    args_parser.add_argument(
        '--config', '-c', default='./configs', metavar='config_path',
        help='dir with toml files or single toml file')
    args_parser.add_argument(
        '--log', '-l', default='./log', metavar='log_dir',
        help='the dir to save log files')
    args_parser.add_argument(
        '--result', '-r', default='./OGNNMDA_result', metavar='result_dir',
        help='the dir to save outputs')
    args = args_parser.parse_args()
    log_file = Path(args.log)
    log_file.mkdir(parents=True, exist_ok=True)
    logger = get_logger('model_train', log_file / 'train.log')
    logger.warning(f'{__file__} started')
    config_path = Path(args.config)
    result_path = Path(args.result)
    logger.info('Preparing for training ...')
    if result_path.is_file():
        logger.error(f'result dir error! :{result_path}')
    else:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        if not config_path.exists():
            logger.error(f'config path error! :{config_path}')
        elif config_path.is_file():
            logger.info(f'using config: {config_path}')
            try:
                train_model(config_path, Path(args.result), logger)
            except Exception as e:
                logger.error(e.args)
                logger.error(traceback.format_exc())
        elif config_path.is_dir():
            config_file_lst = list(filter(lambda file: file.suffix == '.toml', config_path.iterdir()))
            if len(config_file_lst) < 1:
                logger.error(f'config path error! no config file(.toml) in {config_path}')
            for config_file in config_file_lst:
                logger.info(f'using config: {str(config_file)}')
                retry = True
                while retry:
                    retry = False
                    try:
                        train_model(config_file, Path(args.result), logger)
                    except torch.cuda.OutOfMemoryError as e:
                        logger.error(e.args)
                        logger.error(traceback.format_exc())
                        time.sleep(60)
                        logger.info('retry ...')
                        retry = True
                    except Exception as e:
                        logger.error(e.args)
                        logger.error(traceback.format_exc())
    logger.info('training end')
    logger.warning(f'{__file__} exit')
