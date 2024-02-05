import os
import torch
import torch.multiprocessing as mp
import random

from utils import parse_args
from eval import evaluate_kg, cold_start_evaluate
import false_experiment, cold_start_experiment, decrease_experiment, noknowledge_experiment

def run(args_queue:mp.Queue, device:torch.device, lock:mp.Lock):
    cnt = 0
    while True:
        args_dict = args_queue.get()
        if args_dict == None:
            print("{}_cnt:{}    break".format(device, cnt))
            break

        # Environment settings
        experiment      : str   = args_dict['experiment']
        dataset         : str   = args_dict['dataset']
        test_type       : str   = args_dict['test_type']
        rate            : float = args_dict['rate']
        eval_times      : int   = args_dict['eval_times'] 
        model_type_str  : str   = args_dict['model_type_str']
        topk            : list  = args_dict['topk']
        metrics         : list  = args_dict['metrics']
        save_dataset    : bool  = args_dict['save_dataset']
        save_dataloaders: bool  = args_dict['save_dataloaders']

        if experiment == 'false':
            run_once = false_experiment.run_once
        elif experiment == 'coldstart':
            run_once = cold_start_experiment.run_once
        elif experiment == 'decrease':
            run_once = decrease_experiment.run_once
        elif experiment == 'noknowledge':
            run_once = noknowledge_experiment.run_once
        else:
            raise NameError('Invalid experiment.')

        print("{}_cnt:{}    run".format(device, cnt))
        for i in range(eval_times):
            # Mutant dataset generation
            dst_dataset = f'{dataset}-{experiment}-{test_type}-{rate}-{i}'
            if experiment == 'coldstart':
                dst_dataset = f"{dataset}-{experiment}-{test_type}-{rate}-{i}-{args_dict['cs_threshold']}-{args_dict['test_user_ratio']}"
            lock.acquire()
            if not os.path.exists(f'./dataset/{dst_dataset}'):
                os.makedirs(f'./dataset/{dst_dataset}')
                p = mp.Process(target=run_once, args=[args_dict, dst_dataset, i])
                p.start()
                p.join()
            lock.release()

            # Evaluation
            save_root = f'./result/{dataset}/{experiment}_experiment/{test_type}_test'
            if not os.path.exists(save_root):
                try:
                    os.makedirs(save_root)
                except:
                    pass
            save_path = os.path.join(save_root, f'{model_type_str}_{rate}.txt')
            

            config_file =  os.path.join('./config', dataset + "_" + model_type_str + ".yaml")
            if not os.path.exists(config_file):
                config_file = None
            if experiment == 'coldstart':
                save_path = os.path.join(save_root, '{}_{}_{}.txt'.format(model_type_str, rate, args_dict['cs_threshold']))
                p = mp.Process(target=cold_start_evaluate,
                               args=[model_type_str, device, topk, save_path, dst_dataset,
                                     metrics, experiment, config_file, i, save_dataset, save_dataloaders, lock])
            else:
                p = mp.Process(target=evaluate_kg,
                               args=[model_type_str, device, topk, save_path, dst_dataset,
                                     metrics, experiment, config_file, i, save_dataset, save_dataloaders, lock])
            p.start()
            p.join()
        cnt += 1
    return 0

if __name__ == '__main__':
    random.seed(0)
    torch.random.manual_seed(0)
    mp.set_start_method('spawn')
    lock = mp.Lock()
    args = parse_args()
    queue = mp.Queue()
    offset = args.offset

    if args.experiment == 'coldstart':
        con1, con2 = mp.Pipe()
        all_test_users = []
        for i in range(args.eval_times):
            p = mp.Process(target=cold_start_experiment.generate_test_user_list,
                           args=[con1, args.dataset, args.test_user_ratio, torch.device(f'cuda:{offset}'), 0])
            p.start()
            all_test_users.append(con2.recv())
            # print(all_test_users[-1])
            p.join()
        con1.close()
        con2.close()


    for rate in args.rate:
        for test_type in args.test_type_list:
            for model_type_str in args.model:
                if os.path.exists('./result/{}/{}_experiment/{}_test/{}_{}.txt'\
                        .format(args.dataset, args.experiment, test_type, model_type_str, rate)):
                    continue
                if args.experiment == 'coldstart' and os.path.exists('./result/{}/{}_experiment/{}_test/{}_{}_{}.txt'\
                        .format(args.dataset, args.experiment, test_type, model_type_str, rate, args.cs_threshold)):
                    continue
                args_dict = dict()
                args_dict['experiment']             = args.experiment
                args_dict['dataset']                = args.dataset
                args_dict['test_type']              = test_type
                args_dict['rate']                   = rate
                args_dict['model_type_str']         = model_type_str
                args_dict['eval_times']             = args.eval_times
                args_dict['topk']                   = args.topk
                # args_dict['worker_num']             = args.worker_num + offset
                args_dict['metrics']                = args.metrics
                args_dict['save_dataset']           = args.save_dataset
                args_dict['save_dataloaders']       = args.save_dataloaders
                if args.experiment == 'coldstart':
                    args_dict['all_test_users'] = all_test_users
                    args_dict['cs_threshold'] = args.cs_threshold
                    args_dict['test_user_ratio'] = args.test_user_ratio
                queue.put(args_dict)
    for i in range(args.worker_num):
        queue.put(None)
    process_list = []
    for i in range(offset,offset+args.worker_num):
        p = mp.Process(target=run, args=[queue, torch.device('cuda:'+str(i)), lock])
        p.start()
        process_list.append(p)
    for p in process_list:
        p.join()
    print("Finished!")