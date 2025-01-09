import copy

import numpy as np
import torch


def input_evaluation(model, test_x, with_acc=False, use_test=True):
    result = dict()
    if hasattr(model, 'test_loss') and use_test:
        if with_acc:
            test_loss, test_acc = model.test_loss(test_x)
            result['acc'] = test_acc.cpu().detach().numpy()
        else:
            test_loss = model.test_loss(test_x)

    else:
        test_loss = model.loss(test_x)

    result['loss'] = test_loss.cpu().detach().numpy().item()

    return result


def evaluate(model, n_test, with_acc=False, use_test=True, sampler=None, use_grad=True, mc_rounds=1):
    result = dict()
    result['loss'] = 0.
    result['acc'] = 0.
    for n in range(mc_rounds):
        if hasattr(model, 'test_sampler') and use_test:
            test_x = model.test_sampler.sample(n_test)
        elif hasattr(model, 'initializer'):
            test_x = model.initializer.sample(n_test)
        else:
            test_x = sampler.sample(n_test)
        if use_grad:
            output = input_evaluation(model, test_x, with_acc, use_test)
        else:
            with torch.no_grad():
                output = input_evaluation(model, test_x, with_acc, use_test)

        result['loss'] += output['loss']
        if with_acc:
            result['acc'] += output['acc']

    result['loss'] /= mc_rounds
    result['acc'] /= mc_rounds

    return result


def averaged_adam(model, averaged_model, nr_steps, bs, n_test, mode='G', M=1, start=None, gamma=0.9, lr=0.001, eval_steps=50,
                  lr_steps=None, lr_factor=0.2, with_acc=False, use_test_eval=True, lr_exp=None, sampler=None,
                  test_sampler=None, mc_samples=1, eval_grad=False):
    """ Adam optimizer with averaging.
    Two possible options: 'G' (geometric) and 'A' (arithmetic) mean.
    In arithmetic mode, M is the number of previous steps over which the average is computed.
    If M is None, take previous steps starting from step start.
    In geometric mode, gamma is the decay factor (similar as for classical momentum)."""
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    if mode == 'A' and M is not None:
        previous_p_list = [[p.clone().detach() for p in model.parameters()]]
    model.lr = lr
    averaged_model.lr = lr

    errors = []
    accs = []
    train_losses = []

    for n in range(nr_steps):
        optim.zero_grad()
        if hasattr(model, 'initializer'):
            x = model.initializer.sample(bs)
        else:
            x = sampler.sample(bs)
        loss_value = model.loss(x)
        loss_value.backward()
        optim.step()

        loss = loss_value

        if lr_exp is not None:
            model.lr = lr * (n+1) ** (- lr_exp)
            averaged_model.lr = lr * (n+1) ** (- lr_exp)
            for g in optim.param_groups:
                g['lr'] = lr * (n+1) ** (- lr_exp)

        with torch.no_grad():
            # updating the averaged parameters
            if mode == 'A':
                if M is None:
                    if n < start:
                        for p, pp in zip(model.parameters(), averaged_model.parameters()):
                            pp.copy_(p)
                    else:
                        for p, pp in zip(model.parameters(), averaged_model.parameters()):
                            pp.mul_((n+1 - start))
                            pp.add_(p)
                            pp.mul_(1. / (n + 2 - start))
                else:
                    for p, pp, prev in zip(model.parameters(), averaged_model.parameters(), previous_p_list[0]):
                        pp.add_((p - prev) / M)
                    previous_p_list.append([p.clone().detach() for p in model.parameters()])
                    if n >= M - 1:
                        previous_p_list.pop(0)
            elif mode == 'G':
                for p, pp in zip(model.parameters(), averaged_model.parameters()):
                    pp.mul_(gamma)
                    pp.add_((1. - gamma) * p)
            else:
                raise ValueError("Mode must be either 'G' or 'A'.")

        if lr_steps is not None:
            if (n + 1) % lr_steps == 0:
                model.lr *= lr_factor
                averaged_model.lr *= lr_factor
                for g in optim.param_groups:
                    g['lr'] *= lr_factor

        if (n + 1) % eval_steps == 0:
            result = evaluate(averaged_model, n_test, with_acc, use_test_eval, test_sampler, mc_rounds=mc_samples, use_grad=eval_grad)
            if with_acc:
                accs.append(result['acc'].detach().numpy())
            errors.append(result['loss'])
            train_losses.append(loss.detach().numpy())

    output = dict()
    output['errors'] = errors
    output['train_losses'] = train_losses

    if with_acc:
        output['accs'] = accs

    return output


def averaged_adam_multi(model, nr_steps, bs, n_test, gamma_list, factor_list, M=1, lr=0.001, eval_steps=50,
                        lr_steps=None, lr_factor=0.2, with_acc=False, use_test_eval=True, lr_exp=None,
                        lr_decay_start=0, sampler=None, test_sampler=None, eval_grad=True, mc_rounds=1, decay=0.):
    """ Adam optimizer with averaging.
    Running multiple models in parallel with arithmetic average over last k*M steps, for k in the factor_list,
    as well as geometric average with factor gamma, for gamma in the gamma_list."""
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

    model.lr = lr
    averaged_models = [copy.deepcopy(model) for _ in factor_list]
    geo_averaged_models = [copy.deepcopy(model) for _ in gamma_list]

    help_sum_list = [p.clone().detach().zero_() for p in model.parameters()]

    output = dict()
    output['errors'] = dict()
    output['train_losses'] = dict()
    if with_acc:
        output['accs'] = dict()
        output['accs'][0] = []

    for k in factor_list + gamma_list + [0]:
        output['errors'][k] = []
        output['train_losses'][k] = []
        if with_acc:
            output['accs'][k] = []

    k_max = max(factor_list)

    for n in range(nr_steps):
        optim.zero_grad()
        if hasattr(model, 'initializer'):
            x = model.initializer.sample(bs)
        else:
            x = sampler.sample(bs)
        loss_value = model.loss(x)
        loss_value.backward()
        optim.step()

        if lr_exp is not None:
            model.lr = lr * (max(n+1 - lr_decay_start, 1)) ** (- lr_exp)
            for g in optim.param_groups:
                g['lr'] = lr * (max(n+1 - lr_decay_start, 1)) ** (- lr_exp)

        with torch.no_grad():
            for p, q in zip(model.parameters(), help_sum_list):
                q.add_(p)
            # updating the averaged parameters
            if (n + 1) % M == 0:
                for q in help_sum_list:
                    q /= M
                if n + 1 == M:
                    previous_p_list = [[p.clone() for p in help_sum_list] for _ in range(k_max)]
                    for net in averaged_models:
                        for p, q in zip(net.parameters(), help_sum_list):
                            p.copy_(q)

                for k, net in zip(factor_list, averaged_models):
                    for p, pp, prev in zip(help_sum_list, net.parameters(), previous_p_list[-k]):
                        pp.add_((p - prev) / k)

                previous_p_list.append([p.clone() for p in help_sum_list])
                previous_p_list.pop(0)

                for q in help_sum_list:
                    q.zero_()

            for gamma, net in zip(gamma_list, geo_averaged_models):
                for p, pp in zip(model.parameters(), net.parameters()):
                    pp.mul_(gamma)
                    pp.add_((1. - gamma) * p)

        if lr_steps is not None:
            if (n + 1) % lr_steps == 0 and n+1 >= lr_decay_start:
                model.lr *= lr_factor
                for g in optim.param_groups:
                    g['lr'] *= lr_factor

        if (n + 1) % eval_steps == 0:
            print(f'{n + 1} steps completed. Current training loss: {loss_value}.')
            for k, net in zip(factor_list + gamma_list + [0], averaged_models + geo_averaged_models + [model]):

                result = evaluate(net, n_test, with_acc, use_test_eval,
                                  sampler=test_sampler, use_grad=eval_grad, mc_rounds=mc_rounds)
                train_loss = net.loss(x)
                if with_acc:
                    output['accs'][k].append(result['acc'])
                output['errors'][k].append(result['loss'])
                output['train_losses'][k].append(train_loss.cpu().detach().numpy().item())

    return output


def sgd(model, nr_steps, bs, n_test, lr=0.001, eval_steps=50, lr_steps=None, lr_factor=0.2, with_acc=False,
        use_test_eval=True, lr_exp=None, lr_decay_start=0, mc_rounds=1, eval_grad=False, sampler=None, test_sampler=None):
    """Vanilla SGD method for comparison."""

    model.lr = lr

    errors = []
    accs = []
    train_losses = []

    for n in range(nr_steps):

        if hasattr(model, 'initializer'):
            x = model.initializer.sample(bs)
        else:
            x = sampler.sample(bs)
        loss_value = model.loss(x)
        loss_value.backward()
        grads = [p.grad for p in model.parameters()]

        loss = loss_value

        if lr_exp is not None:
            model.lr = lr * (max(n+1 - lr_decay_start, 1)) ** (- lr_exp)

        with torch.no_grad():
            for p, grad in zip(model.parameters(), grads):  # updating the parameters
                p.sub_(model.lr * grad)

        for w in model.parameters():
            w.grad.zero_()

        if lr_steps is not None:
            if (n + 1) % lr_steps == 0:
                model.lr *= lr_factor

        if (n + 1) % eval_steps == 0:
            print(f'{n+1} steps completed. Current training loss: {loss}.')
            result = evaluate(model, n_test, with_acc, use_test_eval, test_sampler, mc_rounds=mc_rounds, use_grad=eval_grad)
            if with_acc:
                accs.append(result['acc'])
            errors.append(result['loss'])
            train_losses.append(loss.cpu().detach().numpy().item())

    output = dict()
    output['errors'] = errors
    output['train_losses'] = train_losses
    if with_acc:
        output['accs'] = accs

    return output


def copy_params(model1, model2):
    with torch.no_grad():
        for p, q in zip(model1.parameters(), model2.parameters()):
            p.copy_(q)


def test_sgd_av_adam(ann, train_steps, eval_steps, bs, n_test, lr, gamma_list, factor_list, M,
                     with_sgd=False, lr_exp=None, lr_decay_start=0, use_test_loss_eval=True, with_acc=False,
                     eval_grad=False, sampler=None, test_sampler=None, mc_rounds=1, decay=0):
    results = dict()
    eval_step_range = np.arange(eval_steps, train_steps + eval_steps, eval_steps)

    if with_sgd:
        copied_model = copy.deepcopy(ann)  # saving parameters for start of next training run
        print('----------------------------------Testing SGD-----------------------------------')

        output = sgd(ann, train_steps, bs, n_test, lr_exp=lr_exp, lr=lr, mc_rounds=mc_rounds, lr_decay_start=lr_decay_start,
                     eval_steps=eval_steps, with_acc=with_acc, use_test_eval=use_test_loss_eval, eval_grad=eval_grad,
                     sampler=sampler, test_sampler=test_sampler)

        test_errors = output['errors']
        train_errors = output['train_losses']

        save_string = f'SGD'

        results[save_string] = dict()

        results[save_string]['errors'] = test_errors
        results[save_string]['train_loss'] = train_errors
        results[save_string]['step_list'] = eval_step_range.tolist()

        if with_acc:
            results[save_string]['accs'] = output['accs']
        copy_params(ann, copied_model)

    print('---------------------------------Testing adam with averaging.------------------------------------')
    output = averaged_adam_multi(ann, train_steps, bs, n_test, gamma_list, factor_list, M, lr, eval_steps=eval_steps,
                                 lr_exp=lr_exp, lr_decay_start=lr_decay_start, eval_grad=eval_grad, sampler=sampler,
                                 test_sampler=test_sampler, with_acc=with_acc, mc_rounds=mc_rounds, decay=decay)
    test_errors = output['errors']
    train_errors = output['train_losses']

    save_string = f'Adam'

    results[save_string] = dict()

    results[save_string]['errors'] = test_errors[0]
    results[save_string]['train_loss'] = train_errors[0]
    results[save_string]['step_list'] = eval_step_range.tolist()
    if with_acc:
        accs = output['accs']
        results[save_string]['accs'] = accs[0]

    for i in range(len(factor_list)):
        k = factor_list[i]
        save_string = f'p{k * M}'

        results[save_string] = dict()

        results[save_string]['errors'] = test_errors[k]
        results[save_string]['train_loss'] = train_errors[k]
        results[save_string]['step_list'] = eval_step_range.tolist()
        if with_acc:
            results[save_string]['accs'] = accs[k]

    for i in range(len(gamma_list)):
        gamma = gamma_list[i]
        save_string = f'g{gamma}'

        results[save_string] = dict()

        results[save_string]['errors'] = test_errors[gamma]
        results[save_string]['train_loss'] = train_errors[gamma]
        results[save_string]['step_list'] = eval_step_range.tolist()
        if with_acc:
            results[save_string]['accs'] = accs[gamma]

    return results