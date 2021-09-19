function [N, centers] = FindCircle(pic , r1, r2, T)
    
    tmp2 = zeros(size(pic,1),size(pic,2));
    for r = r1:r2
        % making circular filter with radious = r 
        F = zeros(2*r+1);
        for theta=1:360
            x=r*cosd(theta);
            y=r*sind(theta);
            F(round(x)+r+1,round(y)+r+1) = 1;
        end
        
        % convolution of filter with picture
        tmp1 = conv2(pic,F,'same');
        tmp2 = max(tmp2 , tmp1);

        surf(tmp1);
        title(['R = ', num2str(r)]);
        pause(1);
    end

%     tmp1 = imregionalmax(tmp2);
%     tmp1 = find(tmp1>0);
%     [tmpx , tmpy] = find(tmp2(tmp1)>T);

%     centers = [tmpx , tmpy];

    [tmpx , tmpy] = find(tmp2>T);
    centers = [tmpy , tmpx];

    surf(tmp2);
    title('Max of output');
    pause(1);
    
    tmpn = 18;
    % now we cluster output to find number of circles
    A = zeros(size(tmp2,1)+2*tmpn, size(tmp2,2)+2*tmpn, 2);
    A (tmpy+tmpn,tmpx+tmpn,1) = 1;
    N = 0;
   
    for i = tmpn+1:size(tmp2,1)
        for j = tmpn+1:size(tmp2,2)
            if(A(i,j,1))
                flag = 0;
                for tmpi = i-tmpn:i+tmpn
                    for tmpj = j-tmpn:j+tmpn
                        if(A(tmpi,tmpj,2) == 1)
                            flag = 1;
                        end
                    end
                end

                if(flag == 0)
                    N = N + 1;
                end                
                A(i,j,2) = 1;
            end
        end
    end
end
