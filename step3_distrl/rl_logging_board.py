r"""Streamlit dashboard for inspecting RL training metrics from local logs."""
import os
import copy
import traceback

try:
    import ujson as json
except:
    import json
    print('`pip install ujson` can be faster.')

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

TOPK = 5
STEP_DATA_KEYS = (
    'prompt',
    'response',
    'ref_response',
    'reward',
    'ref_reward',
    'response_tokens',
    'logprobs',
    'ref_logprobs',
    'probs',
    'ref_probs',
    'values',
    'token_rewards',
    'kl',
    'avg_kl',
    'sum_kl',
    'log_ratio',
    'avg_log_ratio',
    'sum_log_ratio',
    'valid_reward',
    'ref_valid_reward',
    'response_tokens_len',
    'ground_truth',
)
TOPK_FIELD_SUFFIXES = (
    'tokens',
    'advs',
    'advs_normed',
    'policy_logp',
    'ref_logp',
)
FIELD_ALIASES = {
    'values': ('value', 'step_rewards'),
}

st.set_page_config(
    page_title="RL Logging Board",
    page_icon="chart_with_upwards_trend",
    layout='wide'
)


def init_step_logging_data():
    step_data = {key: [] for key in STEP_DATA_KEYS}
    for k in range(TOPK):
        for suffix in TOPK_FIELD_SUFFIXES:
            step_data[f'top{k}_{suffix}'] = []
    return step_data


def get_data_value(data: dict, key: str):
    if key in data:
        return data[key]

    for alias in FIELD_ALIASES.get(key, ()):
        if alias in data:
            return data[alias]

    return None


def has_numeric_data(series_list: list):
    for series in series_list:
        if not series:
            continue
        try:
            if np.isfinite(np.asarray(series, dtype=float)).any():
                return True
        except (TypeError, ValueError):
            return True
    return False


def render_log_text(label: str, content: str):
    st.markdown(label)
    st.code(str(content or ''), language='text', wrap_lines=True)


def plot_histogram_distribution(hist_data: list, group_labels: list, title: str):
    fig = go.Figure()
    colors = px.colors.qualitative.Set2

    for index, (series, label) in enumerate(zip(hist_data, group_labels)):
        series_array = np.asarray(series, dtype=float)
        series_array = series_array[np.isfinite(series_array)]
        if not len(series_array):
            continue

        fig.add_trace(go.Histogram(
            x=series_array,
            name=label,
            opacity=0.65,
            histnorm='probability density',
            marker_color=colors[index % len(colors)],
        ))

    fig.update_layout(
        title=title,
        barmode='overlay',
        legend=dict(orientation='h'),
        xaxis_title='Reward',
        yaxis_title='Density',
    )
    return fig


def load_log_file(
    logdir: os.PathLike,
    max_samples_each_step: int
):
    """
    Parse local JSONL log files.

    Args:
        logdir (os.PathLike): Directory containing JSONL logs.
        max_samples_each_step (int): Maximum samples to keep for each step.
    """
    st.session_state['logging_name'] = logdir
    st.session_state['max_samples_each_step'] = max_samples_each_step
    st.session_state['logging_data'] = {}
    error_lines, success_lines = 0, 0
    
    all_logs = sorted(
        file_name for file_name in os.listdir(logdir)
        if file_name.endswith('.jsonl')
    )
    
    progress_text = f"Processing all files..."
    loading_files_bar = st.progress(0., text=progress_text)

    progress_text = f"Processing each file samples..."
    loading_samples_bar = st.progress(0., text=progress_text)

    if not all_logs:
        st.warning(f'No log file(s) found in {logdir}.')
        st.stop()
    
    
    for log_index, log_name in enumerate(all_logs):
        rl_log_file = os.path.join(logdir, log_name)
        
        mock_max_lines_num = 10000

        with open(rl_log_file, 'r', encoding='utf8', errors='ignore') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    data['step'] = int(data['step'])
                    if data['step'] not in st.session_state['logging_data']:
                        st.session_state['logging_data'][data['step']] = init_step_logging_data()

                    step_data = st.session_state['logging_data'][data['step']]
                    if len(step_data['prompt']) >= max_samples_each_step:
                            percentage = (i + 1) / mock_max_lines_num
                            percentage = min(percentage, 1.0)
                            loading_samples_bar.progress(percentage, text=f"[{int(percentage * 100)}%] Processing {i + 1} / {mock_max_lines_num} samples in each files...")
                            continue
                        
                    for key in STEP_DATA_KEYS:
                        value = get_data_value(data, key)
                        if value is not None:
                            step_data[key].append(value)
                    
                    if 'response_tokens' in data:
                        step_data['response_tokens_len'].append(len(data['response_tokens']))
                        
                    if 'logprobs' in data and 'ref_logprobs' in data:
                        logp = np.array(data['logprobs'])
                        ref_logp = np.array(data['ref_logprobs'])
                        log_ratio = logp - ref_logp
                        kl = np.exp(log_ratio) - 1 - log_ratio
                        step_data['log_ratio'].append(log_ratio.tolist())
                        step_data['avg_log_ratio'].append(np.nanmean(log_ratio))
                        step_data['sum_log_ratio'].append(np.nansum(log_ratio))
                        step_data['kl'].append(kl.tolist())
                        step_data['avg_kl'].append(np.nanmean(kl))
                        step_data['sum_kl'].append(np.nansum(kl))
                        step_data['probs'].append(np.exp(logp).tolist())
                        step_data['ref_probs'].append(np.exp(ref_logp).tolist())

                    for k in range(TOPK):
                        for suffix in TOPK_FIELD_SUFFIXES:
                            field_name = f'top{k}_{suffix}'
                            if field_name in data:
                                step_data[field_name].append(np.array(data[field_name]).tolist())
                    
                    success_lines += 1
                    
                except Exception:
                    print(traceback.format_exc())
                    error_lines += 1
                
                percentage = (i + 1) / mock_max_lines_num
                percentage = min(percentage, 1.0)
                loading_samples_bar.progress(percentage, text=f"[{int(percentage * 100)}%] Processing {i + 1} / {mock_max_lines_num} samples...")

            file_percentage = (log_index + 1) / len(all_logs)
            loading_files_bar.progress(file_percentage, text=f"[{int(file_percentage * 100)}%] Loading {log_index + 1} / {len(all_logs)} files...")

    percentage = 1.0
    loading_samples_bar.progress(percentage, text=f"[{int(percentage * 100)}%] Processing {(success_lines + error_lines)} / {(success_lines + error_lines)} samples...")
    
    st.toast(
        f'Loaded {success_lines + error_lines} sample(s), sucess: {success_lines}, error: {error_lines}.'
    )

    if not st.session_state['logging_data']:
        st.warning(f'No log file(s) found in {logdir}.')
        st.stop()

    all_steps = [int(s) for s in list(st.session_state["logging_data"].keys())]
    all_steps.sort()
    st.session_state['max_step_index'] = max(all_steps)
    st.session_state['min_step_index'] = min(all_steps)
    st.session_state['step_gap'] = 1 if len(all_steps) < 2 else all_steps[1] - all_steps[0]
    
    rewards_dict = {'step': [], 'reward': [], 'ref_reward': []}
    for step in st.session_state['logging_data']:
        st.session_state['logging_data'][step]['avg_reward'] = sum(st.session_state['logging_data'][step]['reward']) / len(st.session_state['logging_data'][step]['reward'])
        
        current_step_resp_length = [len(resp) for resp in st.session_state['logging_data'][step]['response']]
        st.session_state['logging_data'][step]['avg_length'] = int(sum(current_step_resp_length) / len(current_step_resp_length))
        
        current_step_ref_resp_length = [len(resp) for resp in st.session_state['logging_data'][step]['ref_response']]
        st.session_state['logging_data'][step]['avg_ref_length'] = int(sum(current_step_ref_resp_length) / len(current_step_ref_resp_length)) if len(current_step_ref_resp_length) else 0
        
        if len(st.session_state['logging_data'][step]['ref_reward']):
            st.session_state['logging_data'][step]['avg_ref_reward'] = sum(st.session_state['logging_data'][step]['ref_reward']) / len(st.session_state['logging_data'][step]['ref_reward']) if len(st.session_state['logging_data'][step]['ref_reward']) else 0
        else:
            st.session_state['logging_data'][step]['avg_ref_reward'] = 0
        rewards_dict['step'].append(step)
        rewards_dict['reward'].append(st.session_state['logging_data'][step]['avg_reward'])
        rewards_dict['ref_reward'].append(st.session_state['logging_data'][step]['avg_ref_reward'])
    
    rewards_df = pd.DataFrame.from_dict(rewards_dict)
    st.session_state['reward_df'] = rewards_df.set_index('step')


def init_sidebar():
    """
    Initialize sidebar controls.
    """
    st.sidebar.markdown(
        "<h1 style='text-align: center;'>RL Logging Board</h1>",
        unsafe_allow_html=True
    )

    base_root_path = st.sidebar.text_input(
        "Log(s) Root Path",
        value='./rollout_samples',
    )
    
    if not os.path.exists(base_root_path):
        st.warning(f'Log(s) Root Path: `{base_root_path}` does not exist.')
        st.stop()
    
    all_log_path_in_logdir = os.listdir(base_root_path)
    
    if not all_log_path_in_logdir:
        st.warning('No log files found.')
        st.code("""Logging Dir should be like:  
Base Log Dir  
    |__eval_topk_0_topp_1 (dir for evaluate logs)   
    |   |__eval.jsonl  
    |__topk_0_topp_1 (dir for training logs, only for rl logs)  
        |__rollout_data_rank_0_1313.jsonl  
    ...   
""")
        st.stop()

    log_name = st.sidebar.selectbox(
        'Choose Log Name',
        options=all_log_path_in_logdir,
        index=len(all_log_path_in_logdir) - 1
    )
    
    max_samples_each_step = st.sidebar.number_input(
        'Max Samples Each Step',
        help='Downsample each step when large batches make the dashboard slow.',
        value=128,
        max_value=10240,
        min_value=1
    )
    
    load_btn = st.sidebar.button(
        "Load & View",
        use_container_width=True
    )
    
    if load_btn and (
        'logging_data' not in st.session_state 
        or 
        log_name != st.session_state['logging_name']
        or
        max_samples_each_step != st.session_state.get('max_samples_each_step', -1)
    ):
        load_log_file(
            os.path.join(base_root_path, log_name), 
            max_samples_each_step
        )
    
    with st.sidebar.expander('Module settings', expanded=True):
        st.session_state['show_reward_logging'] = st.checkbox('Reward curves', value=True)
        st.session_state['var_scaling'] = st.slider('Variance Scaling', min_value=0.1, max_value=1.0, value=0.2, help='Scale the shaded variance area in reward curves.')
        st.session_state['zero_shift'] = st.checkbox('Zero Shift', value=False, help='Shift the first value of each reward curve to 0 for trend comparison only.')
        st.session_state['show_response'] = st.checkbox('Response comparison', value=True)

    with st.sidebar.expander('Detail settings', expanded=True):
        st.session_state['use_logp_as_kl'] = st.checkbox('Use LogP as KL', value=True, help='Show LogProb instead of KL in reward curves.')
        st.session_state['drop_pad'] = st.checkbox('Drop Padding Token', value=True)
        st.session_state['pad_token'] = st.text_input('Pad Token', value='<PAD>', disabled=not st.session_state['drop_pad'])
        st.session_state['drop_sys_prompt'] = st.checkbox('Drop System Prompt', value=True)
        st.session_state['end_token_of_sys_prompt'] = st.text_input('End Token of System Prompt', value='<endofsystem>', disabled=not st.session_state['drop_sys_prompt'])
        st.session_state['show_charts'] = st.checkbox('Show Charts', value=True)
        st.session_state['show_batch_samples'] = st.checkbox('Show Batch Samples', value=True)
        st.session_state['show_samples_pair'] = st.checkbox('Show Samples Pair', value=True)
        st.session_state['show_token_heat_map'] = st.checkbox('Show Heat Map', value=True)

def plot_filled_line(
    x: list,
    y_list_list: list,
    data_names: list,
    colors: list,
    title=None,
    var_scaling=1.
):
    """
    Plot a line chart with a min/max shaded band for each x value.

    Args:
        x (list): Step indices on the x axis.
        y_list_list (line_num, steps, step_wise): Per-line values for each step.
        data_names (list): Line names.
        colors (list): RGB color strings, e.g. ['255,171,171'].
    """
    fig = go.Figure()
    
    x_rev = x[::-1]
    for i in range(len(y_list_list)):
        y_list = y_list_list[i]
        zero_shift_value = 0
        y_mean, y_lower, y_upper = [], [], []
        
        for idx, y in enumerate(y_list):
            y_arr = np.asarray(y, dtype=float)
            if not np.isfinite(y_arr).any():
                y_mean.append(np.nan)
                y_lower.append(np.nan)
                y_upper.append(np.nan)
                continue

            if idx == 0 and st.session_state.get('zero_shift', False):
                zero_shift_value = np.nanmean(y_arr)
            
            y_arr = y_arr - zero_shift_value
            mean, std = float(np.nanmean(y_arr)), float(np.nanstd(y_arr))
            std *= var_scaling
            y_mean.append(mean)
            y_lower.append(mean - std)
            y_upper.append(mean + std)
        y_lower = y_lower[::-1]

        fig.add_trace(go.Scatter(
            x=x + x_rev,
            y=y_upper + y_lower,
            fill='toself',
            fillcolor=f'rgba({colors[i]},0.1)',
            line_color='rgba(255,255,255,0)',
            showlegend=False,
            name=data_names[i],
        ))
        fig.add_trace(go.Scatter(
            x=x, y=y_mean,
            line_color=f'rgb({colors[i]})',
            name=data_names[i],
        ))

    fig.update_traces(mode='lines')
    
    if title:
        fig.update_layout(
            title=title,
            legend=dict(orientation="h")
        )

    return fig


def main_page():
    """
    Metrics Page.
    """
    if "logging_data" not in st.session_state:
        st.info("Please press the Load & View button to load logs.")
    else:
        if st.session_state['show_reward_logging']:
            step_reward_tab, step_kl_tab, resp_len_tab = st.tabs([
                'Step-Reward', 
                'Step-KL', 
                'Step-RespLen'
            ])
            
            with step_reward_tab:
                steps = sorted(st.session_state['logging_data'])
                reward, ref_reward, valid_reward, ref_valid_reward = [], [], [], []
                for step in steps:
                    value_dict = st.session_state['logging_data'][step]
                    reward.append(value_dict['reward'] or [np.nan])
                    ref_reward.append(value_dict['ref_reward'] or [np.nan])
                    valid_reward.append(value_dict['valid_reward'] or [np.nan])
                    ref_valid_reward.append(value_dict['ref_valid_reward'] or [np.nan])
                
                all_curves = {
                    'ref_reward': {
                        'value': ref_reward,
                        'color': '132,201,255'
                    },
                    'reward': {
                        'value': reward,
                        'color': '255,171,171'
                    }, 
                    'ref_valid_reward': {
                        'value': ref_valid_reward,
                        'color': '132,155,200'
                    }, 
                    'valid_reward': {
                        'value': valid_reward,
                        'color': '200,155,200'
                    }
                }
                
                candidate_curves = [
                    key for key in all_curves
                    if has_numeric_data(all_curves[key]['value'])
                ]
                
                show_curves = st.multiselect(
                    'Show Rewards',
                    candidate_curves,
                    candidate_curves,
                    label_visibility='collapsed'
                )
                
                reward_fig = plot_filled_line(
                    x=steps,
                    y_list_list=[all_curves[r]['value'] for r in show_curves],
                    data_names=show_curves,
                    colors=[all_curves[r]['color'] for r in show_curves],
                    title='Rewards Logging (Step level)',
                    var_scaling=st.session_state['var_scaling']
                )

                st.plotly_chart(reward_fig, theme="streamlit", use_container_width=True)

            with step_kl_tab:
                steps, kl = [], []

                if st.session_state['use_logp_as_kl']:
                    for step, value_dict in st.session_state['logging_data'].items():
                        if value_dict['avg_log_ratio']:
                            steps.append(step)
                            kl.append(value_dict['avg_log_ratio'])
                else:
                    for step, value_dict in st.session_state['logging_data'].items():
                        if value_dict['avg_kl']:
                            steps.append(step)
                            kl.append(value_dict['avg_kl'])
                
                reward_fig = plot_filled_line(
                    x=steps,
                    y_list_list=[kl],
                    data_names=['KL'],
                    colors=['255,165,0'],
                    title='KL Logging (Step level)'
                )
                st.plotly_chart(reward_fig, theme="streamlit", use_container_width=True)
            
            with resp_len_tab:
                steps, resp_len = [], []

                for step, value_dict in st.session_state['logging_data'].items():
                    if value_dict['response_tokens_len']:
                        steps.append(step)
                        resp_len.append(value_dict['response_tokens_len'])

                resp_len_fig = plot_filled_line(
                    x=steps,
                    y_list_list=[resp_len],
                    data_names=['resp_len'],
                    colors=['255,165,0'],
                    title='Response Length Logging (Step level)'
                )
                st.plotly_chart(resp_len_fig, theme="streamlit", use_container_width=True)
        
        if st.session_state['show_response']:
            st.markdown('**Each Step Response**')
            
            if st.session_state['min_step_index'] == st.session_state['max_step_index']:
                step_index = st.session_state['min_step_index']
            elif (
                len(st.session_state['logging_data']) > 2 
                and 
                list(st.session_state['logging_data'].keys())[2] - list(st.session_state['logging_data'].keys())[1] != list(st.session_state['logging_data'].keys())[1] - list(st.session_state['logging_data'].keys())[0]
            ):
                step_index = st.selectbox(
                    f"Step Index({st.session_state['max_step_index']} total steps):",
                    list(st.session_state['logging_data'].keys()),
                    index=0
                )
            else:
                step_index = st.slider(
                    f"Step Index({st.session_state['max_step_index']} total steps):",
                    min_value=st.session_state['min_step_index'],
                    max_value=st.session_state['max_step_index'],
                    value=st.session_state['min_step_index'],
                    step=st.session_state['step_gap']
                )

            cur_step_content_dict = st.session_state['logging_data'][step_index]
            cur_step_filtered_content_dict = copy.deepcopy(cur_step_content_dict)
            
            cur_step_filtered_content_dict['prompt'] = []
            for prompt in cur_step_content_dict['prompt']:
                if st.session_state['drop_pad']:
                    prompt = prompt.replace(st.session_state['pad_token'], '').strip()
                if st.session_state['drop_sys_prompt']:
                    prompt = prompt.split(st.session_state['end_token_of_sys_prompt'])[-1]
                cur_step_filtered_content_dict['prompt'].append(prompt)

            cur_step_filtered_content_dict['response'] = [c.replace(st.session_state['pad_token'], '').strip() if st.session_state['drop_pad'] else c for c in cur_step_content_dict['response']]
            cur_step_filtered_content_dict['reward_gap'] = [r - ref_r for r, ref_r in zip(cur_step_content_dict['reward'], cur_step_content_dict['ref_reward'])]
            cur_step_filtered_content_dict['valid_reward_gap'] = [r - ref_r for r, ref_r in zip(cur_step_content_dict['reward'], cur_step_content_dict['valid_reward'])]
            
            if st.session_state['show_charts']:

                if not cur_step_filtered_content_dict['ref_reward']:
                    cur_step_filtered_content_dict['ref_reward'] = [0] * len(cur_step_filtered_content_dict['reward'])

                c1, c2, c3 = st.columns([6, 6, 6])

                with c1:                                                    # reward distribution
                    reward_distribution_dict = {
                        'sample_index': [],
                        'reward': [],
                        'tag': []
                    }
                    for sample_index, (reward, ref_reward) in enumerate(zip(cur_step_filtered_content_dict['reward'], cur_step_filtered_content_dict['ref_reward'])):
                        reward_distribution_dict['sample_index'].append(sample_index)
                        reward_distribution_dict['reward'].append(reward)
                        reward_distribution_dict['tag'].append('Reward')
                        reward_distribution_dict['sample_index'].append(sample_index)
                        reward_distribution_dict['reward'].append(ref_reward)
                        reward_distribution_dict['tag'].append('Ref Reward')

                    reward_distribution_df = pd.DataFrame.from_dict(reward_distribution_dict)
                    fig = px.bar(
                        reward_distribution_df, 
                        x="sample_index", 
                        y="reward", 
                        color="tag",
                        barmode='group',
                        color_discrete_sequence=px.colors.diverging.Portland,
                        title="Reward in current batch samples"
                    )
                    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                
                with c2:                                                    # reward gap distribution
                    reward_distribution_dict = {
                        'sample_index': [i for i in range(len(cur_step_filtered_content_dict['reward_gap']))],
                        'reward_gap': cur_step_filtered_content_dict['reward_gap']
                    }
                    reward_distribution_df = pd.DataFrame.from_dict(reward_distribution_dict)
                    fig = px.bar(
                        reward_distribution_df, 
                        x="sample_index", 
                        y="reward_gap", 
                        color="reward_gap", 
                        color_discrete_sequence=['red'],
                        title="Reward Gap (r - ref_r) in current batch"
                    )
                    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

                with c3:                                                    # reward variance distribution
                    if cur_step_filtered_content_dict['ref_reward']:
                        hist_data = [
                            cur_step_filtered_content_dict['ref_reward'],
                            cur_step_filtered_content_dict['reward'],
                        ]
                        group_labels = ['Ref Rewards', 'Rewards']
                    else:
                        hist_data = [cur_step_filtered_content_dict['reward']]
                        group_labels = ['Rewards']

                    fig = plot_histogram_distribution(
                        hist_data,
                        group_labels,
                        title="Reward Distribution in current batch"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            showed_keys = [
                'prompt', 
                'response', 
                'reward', 
                'ground_truth',
                'valid_reward', 
                'avg_log_ratio', 
                'sum_log_ratio', 
                'avg_kl', 
                'sum_kl', 
                'ref_response', 
                'ref_reward', 
                'ref_valid_reward', 
                'reward_gap', 
                'valid_reward_gap'
            ]
            candidate_keys = [k for k in showed_keys if cur_step_filtered_content_dict[k]]
            content_dict = dict([(k, cur_step_filtered_content_dict[k]) for k in candidate_keys])
            content_df = pd.DataFrame.from_dict(content_dict)
            
            if st.session_state['show_batch_samples']:
                st.dataframe(
                    content_df, 
                    use_container_width=True,
                    height=350
                )

            if st.session_state['show_samples_pair']:    
                
                c1, c2, c3 = st.columns([1, 1, 4])
                with c1:
                    if step_index == st.session_state['min_step_index']:
                        delta_char = 0
                    else:
                        try:
                            cur_avg_len = st.session_state['logging_data'][step_index]['avg_length']
                            last_avg_len = st.session_state['logging_data'][step_index-st.session_state['step_gap']]['avg_length']
                            delta_char = cur_avg_len - last_avg_len
                        except:
                            delta_char = 0
                    st.metric(                                                  # actor average response length and step-over-step delta
                        'Response Average Length',
                        value=f"{st.session_state['logging_data'][step_index]['avg_length']} chars",
                        delta=f'{delta_char} chars'
                    )
                
                with c2:                                                        # reference model average response length and delta
                    try:
                        delta_char = 0 if step_index == st.session_state['min_step_index'] else st.session_state['logging_data'][step_index]['avg_ref_length'] - st.session_state['logging_data'][step_index-st.session_state['step_gap']]['avg_ref_length']
                    except:
                        delta_char = 0
                    st.metric(
                        'Ref Response Average Length',
                        value=f"{st.session_state['logging_data'][step_index]['avg_ref_length']} chars",
                        delta=f'{delta_char} chars'
                    )
                
                with c3:
                    sample_index = st.number_input(
                        f'Sample index in current step batch: ', 
                        min_value=0,
                        max_value=len(cur_step_filtered_content_dict['response']) - 1,
                        value=0
                    )
                
                # Show one response/ref_response pair from the current step.
                c1, c2, c3, c4 = st.columns([4, 4, 4, 2])
                with c1:
                    content = cur_step_filtered_content_dict["prompt"][sample_index]
                    render_log_text(':gray[Prompt]', content)
                with c2:
                    content = cur_step_filtered_content_dict["response"][sample_index]
                    render_log_text(':green[Response]', content)
                with c3:
                    if (
                            "ref_response" in cur_step_filtered_content_dict
                            and
                            cur_step_filtered_content_dict["ref_response"]
                    ):
                        content = cur_step_filtered_content_dict["ref_response"][sample_index]
                        render_log_text(':blue[Ref Response]', content)
                    else:
                        st.info('No `ref_response` found in log line data.')
                with c4:
                    st.markdown(':orange[Reward Gap]')
                    reward_gap = round(cur_step_filtered_content_dict["reward_gap"][sample_index], 4) if cur_step_filtered_content_dict["reward_gap"] else 0.
                    st.metric(
                        ' ', 
                        value=reward_gap
                    )

                # Show detailed token-level values for the selected sample.
                if 'token_rewards' in cur_step_filtered_content_dict and cur_step_filtered_content_dict['token_rewards']:
                    # Keep token and logprob arrays aligned before rendering heat maps.
                    resp_token_len = len(cur_step_filtered_content_dict['response_tokens'][sample_index])
                    logp_len = len(cur_step_filtered_content_dict['logprobs'][sample_index])
                    if resp_token_len != logp_len:
                        st.info(
                            f'Note: `resp_tokens` (len: {resp_token_len}) is not equal to `logprobs` (len: {logp_len}), this may caused by <PAD> tokens, CLIP response tokens!',
                        )
                        cur_step_filtered_content_dict['response_tokens'][sample_index] = cur_step_filtered_content_dict['response_tokens'][sample_index][:logp_len]

                    topk_show_values = []
                    for k in range(TOPK):
                        topk_show_values.extend([f'top{k}_tokens', f'top{k}_advs', f'top{k}_advs_normed', f'top{k}_policy_logp', f'top{k}_ref_logp'])
                    
                    show_values = st.multiselect(
                        'Select show value(s)',
                        ['token_reward', 'log_ratio', 'kl', 'token_value', 'logp', 'ref_logp', 'prob', 'ref_prob'] + topk_show_values,
                        ['token_reward', 'log_ratio', 'kl', 'token_value', 'logp', 'ref_logp', 'prob', 'ref_prob'] + topk_show_values
                    )
                    
                    new_dict, index_list = {}, []
                    
                    if st.session_state['drop_pad'] and cur_step_filtered_content_dict['response_tokens'][sample_index][-1] == st.session_state['pad_token']:
                        first_pad_token_idx = cur_step_filtered_content_dict['response_tokens'][sample_index].index(st.session_state['pad_token'])
                        response_tokens_without_pad_token = cur_step_filtered_content_dict['response_tokens'][sample_index][:first_pad_token_idx]
                    else:
                        response_tokens_without_pad_token = cur_step_filtered_content_dict['response_tokens'][sample_index]
                    
                    for token_idx in range(len(response_tokens_without_pad_token)):
                        if cur_step_filtered_content_dict['response_tokens']:
                            resp_token = cur_step_filtered_content_dict['response_tokens'][sample_index][token_idx]
                            resp_token = f'{token_idx} - {resp_token}'
                            if resp_token not in new_dict:
                                new_dict[resp_token] = []

                        for k in range(TOPK):
                            if cur_step_filtered_content_dict[f'top{k}_tokens']:
                                topk_token = cur_step_filtered_content_dict[f'top{k}_tokens'][sample_index][token_idx]
                                topk_token = f'{token_idx} - {topk_token}'
                                if f'top{k}_tokens' not in new_dict:
                                    index_list.append(f'top{k}_tokens')

                            if cur_step_filtered_content_dict[f'top{k}_advs']:
                                topk_advs = cur_step_filtered_content_dict[f'top{k}_advs'][sample_index][token_idx]
                                if f'top{k}_advs' in show_values:
                                    new_dict[resp_token].append(topk_advs)
                                    if f'top{k}_advs' not in index_list:
                                        index_list.append(f'top{k}_advs')

                            if cur_step_filtered_content_dict[f'top{k}_advs_normed']:
                                topk_advs_normed = cur_step_filtered_content_dict[f'top{k}_advs_normed'][sample_index][token_idx]
                                if f'top{k}_advs_normed' in show_values:
                                    new_dict[resp_token].append(topk_advs_normed)
                                    if f'top{k}_advs_normed' not in index_list:
                                        index_list.append(f'top{k}_advs_normed')

                            if cur_step_filtered_content_dict[f'top{k}_policy_logp']:
                                topk_policy_logp = cur_step_filtered_content_dict[f'top{k}_policy_logp'][sample_index][token_idx]
                                if f'top{k}_policy_logp' in show_values:
                                    new_dict[resp_token].append(topk_policy_logp)
                                    if f'top{k}_policy_logp' not in index_list:
                                        index_list.append(f'top{k}_policy_logp')

                            if cur_step_filtered_content_dict[f'top{k}_ref_logp']:
                                topk_ref_logp = cur_step_filtered_content_dict[f'top{k}_ref_logp'][sample_index][token_idx]
                                if f'top{k}_ref_logp' in show_values:
                                    new_dict[resp_token].append(topk_ref_logp)
                                    if f'top{k}_ref_logp' not in index_list:
                                        index_list.append(f'top{k}_ref_logp')
                                
                        if cur_step_filtered_content_dict['token_rewards']:
                            token_reward = cur_step_filtered_content_dict['token_rewards'][sample_index][token_idx]
                            if 'token_reward' in show_values:
                                new_dict[resp_token].append(token_reward)
                                if 'token_reward' not in index_list:
                                    index_list.append('token_reward')
                        
                        if cur_step_filtered_content_dict['log_ratio']:
                            log_ratio = cur_step_filtered_content_dict['log_ratio'][sample_index][token_idx]
                            if 'log_ratio' in show_values:
                                new_dict[resp_token].append(log_ratio)
                                if 'log_ratio' not in index_list:
                                    index_list.append('log_ratio')
                        
                        if cur_step_filtered_content_dict['kl']:
                            kl = cur_step_filtered_content_dict['kl'][sample_index][token_idx]
                            if 'kl' in show_values:
                                new_dict[resp_token].append(kl)
                                if 'kl' not in index_list:
                                    index_list.append('kl')
                        
                        if cur_step_filtered_content_dict['values']:
                            value = cur_step_filtered_content_dict['values'][sample_index][token_idx]
                            if 'token_value' in show_values:
                                new_dict[resp_token].append(value)
                                if 'token_value' not in index_list:
                                    index_list.append('token_value')
                        
                        if cur_step_filtered_content_dict['logprobs']:
                            logp = cur_step_filtered_content_dict['logprobs'][sample_index][token_idx]
                            if 'logp' in show_values:
                                new_dict[resp_token].append(logp)
                                if 'logp' not in index_list:
                                    index_list.append('logp')

                        if cur_step_filtered_content_dict['ref_logprobs']:
                            ref_logp = cur_step_filtered_content_dict['ref_logprobs'][sample_index][token_idx]
                            if 'ref_logp' in show_values:
                                new_dict[resp_token].append(ref_logp)
                                if 'ref_logp' not in index_list:
                                    index_list.append('ref_logp')
                        
                        if cur_step_filtered_content_dict['probs']:
                            prob = cur_step_filtered_content_dict['probs'][sample_index][token_idx]
                            if 'prob' in show_values:
                                new_dict[resp_token].append(prob)
                                if 'prob' not in index_list:
                                    index_list.append('prob')
                        
                        if cur_step_filtered_content_dict['ref_probs']:
                            ref_prob = cur_step_filtered_content_dict['ref_probs'][sample_index][token_idx]
                            if 'ref_prob' in show_values:
                                new_dict[resp_token].append(ref_prob)
                                if 'ref_prob' not in index_list:
                                    index_list.append('ref_prob')
                                    
                    try:
                        token_level_df = pd.DataFrame.from_dict(new_dict)
                        renamed_index_dict = dict((i, name) for i, name in enumerate(index_list))
                        token_level_df.rename(
                            index=renamed_index_dict, 
                            inplace=True
                        )
                        
                        st.dataframe(
                            token_level_df.style.background_gradient(axis=1, cmap="binary"), 
                            use_container_width=True
                        )
                        
                        if st.session_state['show_token_heat_map']:
                            fig = px.imshow(
                                token_level_df, 
                                text_auto=True,
                                aspect="auto",
                                color_continuous_scale="balance",
                            )
                            fig.update_xaxes(side="top")
                            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                    except Exception as e:
                        st.error(f'Error occured: {e}.')
                        st.write(new_dict)


if __name__ == '__main__':
    init_sidebar()
    main_page()
