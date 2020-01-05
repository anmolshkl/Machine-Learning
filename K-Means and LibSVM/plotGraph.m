function [] = plotGraph(kList, sosList, p1List, p2List, p3List)
    figure
    plot(kList, sosList);
    title('Total within group sum of squares vs k');
    xlabel('k');
    ylabel('Total within group sum of squares');

    figure
    lp1 = plot(kList, p1List, '-r', 'LineWidth',2);
    hold on;
    lp2 = plot(kList, p2List, '-g', 'LineWidth',2);
    hold on;
    lp3 = plot(kList, p3List, '-b', 'LineWidth',2);
    hold on;

    legend([lp1; lp2; lp3], ["P1", "P2", "P3"]);
    title('P Values vs k');
    xlabel('k');
    ylabel('P Values');
end