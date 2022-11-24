function plotSimu(signal, b, fitNNLS)

    DValues =  fitNNLS(1,:)
    s(:,:) =  fitNNLS(2,:)
    s2(:,:) =  fitNNLS(3,:)
    DBasis = exp(-kron(b', DValues))
    #sNLLS = exp(-kron(b, dNLLS(1)))*fNNLS(1) + exp(-kron(b, dNLLS(2)))*fNNLS(2) + exp(-kron(b, dNLLS(3)))*fNNLS(3)
    
    figure(1)
    singlePlot(signal, DBasis, b, DValues, s, s2)

end

function singlePlot(signal, DBasis, b, DValues, s, s2)
    close all

    subplot(3,1,1)
    y_recon = DBasis*s'
    y_recon2 = DBasis*s2'
    plot( b , signal , 'ko' , b , y_recon , 'b-', b, y_recon2 , 'r-') # compare signal and fitting result
    ylim([0 signal(1)])
    title('(a) Signal decay fitting')
    xlabel('b-values (s/mm^2)')
    ylabel('Signal amplitude')
    xlim([0, b(end)])
    legend('data (a), D^{in}_i (c)','lsqnonneg','NNLSreg')

    subplot(3,1,2)
    plot(b, signal-signal, 'k-', b, signal'-y_recon , 'b-', b, signal'-y_recon2 , 'r-')
    title('(b) Residual plot')
    a=[abs(mean(signal'-y_recon)) abs(mean(signal'-y_recon2))]
    xlim([0, b(end)])
    xlabel(['Avg residual: lsqnonneg = ' num2str(a(1)) ' | NNLSreg = ' num2str(a(2))])
    
    subplot(3,1,3)
    yyaxis left
    semilogx(DValues,s)
    y=ylim
    ytext=y(2)-0.05
    text(1e-3,ytext,'D_{slow}')
    text(4*1e-3,ytext,'D_{inter}')
    text(90*1e-3,ytext,'D_{fast}')
    h(1)=rectangle('Position',[2*1e-3 0.001 8*1e-3 ytext+0.05],'FaceColor',[.9 .9 .9],'EdgeColor','none')
    uistack(h(1),'bottom')
    title('(c) Results')
    xlabel('D (mm^2/s) (\cdot 10^{-3})')
    xlim([0.9*1e-3, 200*1e-3])
    ylabel('Amplitude')
    yyaxis right
    semilogx(DValues,s2,'r-')
    ax = gca
    ax.YAxis(1).Color = 'k'
    ax.YAxis(2).Color = 'r'
    # legend({'lsqnonneg','NNLSreg','D^{in}_i'},'Location','northwest')
end