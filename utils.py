import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_return_over_episodes(episode_axis, return_axis, label):
    plt.title("Total return over episodes")
    plt.xlabel('Episode') 
    plt.ylabel('Total return')
    plt.plot(episode_axis, return_axis, label=label)
    plt.legend()
    plt.savefig("evaluation/training_total_return.png", dpi=300, bbox_inches="tight")
    plt.clf()

def plot_sharpe_ratio_over_episodes(episode_axis, sharpe_ratio_axis, label):
    plt.title("Sharpe Ratio over episodes")
    plt.xlabel('Episode') 
    plt.ylabel('Sharpe Ratio')   
    plt.plot(episode_axis, sharpe_ratio_axis, label=label)
    plt.legend()
    plt.savefig("evaluation/training_sharpe_ratio.png", dpi=300, bbox_inches="tight")
    plt.clf()

def plot_mean_actor_loss_over_episodes(episode_axis, actor_loss_axis, label):
    plt.title("Mean Actor Loss over episodes")
    plt.xlabel('Episode') 
    plt.ylabel('Mean Actor Loss')
    plt.plot(episode_axis, actor_loss_axis, label=label)
    plt.legend()
    plt.savefig("evaluation/training_actor_loss.png", dpi=300, bbox_inches="tight")
    plt.clf()

def plot_mean_critic_loss_over_episodes(episode_axis, critic_loss_axis, label):
    plt.title("Mean Critic Loss over episodes")
    plt.xlabel('Episode') 
    plt.ylabel('Mean Critic Loss')
    plt.plot(episode_axis, critic_loss_axis, label=label)
    plt.legend()
    plt.savefig("evaluation/training_critic_loss.png", dpi=300, bbox_inches="tight")
    plt.clf()

def plot_testing_return(env, testing_modes, return_history):
    plt.title("Cumulative return over time")
    plt.xlabel('Date') 
    plt.ylabel('Cumulative return')
    date_axis = env.trading_date_range()
    plt.xticks(rotation=45)
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=15))
    for mode in testing_modes:
        if (testing_modes[mode] == 1):
            plt.plot(date_axis, return_history[mode], label=mode)
    plt.legend()
    plt.savefig("evaluation/testing_cumulative_return.png", dpi=300, bbox_inches="tight")
    plt.clf()

def print_eval_results(env, total_return):
    sharpe_ratio = env.sharpe_ratio()
    omega_ratio = env.omega_ratio(15)
    mdd = env.maximum_drawdown()
    portfolio_value = env.total_portfolio_value()
    avg_yearly_return = env.avg_yearly_return()
    avg_monthly_return = env.avg_monthly_return()
    print(f"------Portfolio Value {portfolio_value:.2f}; Total Return {total_return:.2f};------")
    print(f"------Sharpe Ratio {sharpe_ratio:.5f}; Omega Ratio {omega_ratio:.5f} MDD {mdd:.5f}------")
    print(f"------Average yearly return {avg_yearly_return:.5f}%; Average monthly return {avg_monthly_return:.5f}%------\n")