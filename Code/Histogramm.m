clear 'all';
close 'all';

%read image
Image = imread('test.jpg');

%plot the image
figure(1);
subplot(2,2,1);
imshow(Image);
title('Birdview');

%get histogram values
[Count, Value] = imhist(Image);
%plot them
subplot(2,2,2);
plot(Value, Count, 'bo-');
xlabel('gray value');
ylabel('absolute frequency')

%plot the relative frequency
subplot(2,2,3);
plot(Value, Count/sum(Count), 'bo-');
xlabel('gray value');
ylabel('relative frequency')

%now determine the cumulative sum
SumCount = cumsum(Count/numel(Image));
%plot the cumulative sum
subplot(2,2,4);
plot(Value, SumCount, 'bo-');
xlabel('gray value');
ylabel('cumulative frequency')
