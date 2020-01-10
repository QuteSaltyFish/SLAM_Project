x=[5 10 13 20];
for n=1:4
    [z,p,k] = buttap(x(n));
    [num,den] = zp2tf(z,p,k);
    [H,w] = freqz(num,den,[0:2*pi/1e3:2*pi]);
    subplot(4,1,n);
    Hf = abs(H);
    Hx = angle(H);
    plot(w,-20*log(Hf));
    set(gca,'fontsize',9,'fontname','Times');
    set(gca, 'XLim',[0 2*pi]); % X轴的数据显示范围 
    set(gca,'XTick',[0:pi/4:2*pi]); % X轴的记号点 
    set(gca,'XTickLabel',{'0' '1/4\pi' '1/2\pi' '3/4\pi' '\pi' '5/4\pi' '3/2\pi' '7/4\pi' '2\pi'}) % X轴的记号
    title(['butterworth filter(',num2str(x(n)),'-order)']);
end
img =gcf;  %获取当前画图的句柄
print(img, '-dpng', '-r900', './img.png')