function Y = Median_Filter( X , Ks )
    F = 1/(Ks^2) * ones(Ks);
    [s1 ,s2] = size(X);
    A = zeros(s1 + Ks - 1, s2 + Ks - 1);
    Y = zeros(s1 + Ks - 1, s2 + Ks - 1);
    
    % try and catch to check if Ks is odd. Implementing filter for even
    % kernels have different definitions and it's not useful and we may get
    % shift in picture
    try
        
    A(1+floor(Ks/2):s1+floor(Ks/2),1+floor(Ks/2):s2+floor(Ks/2)) = X;
    
    for k = 1+floor(Ks/2):s1+floor(Ks/2)
        for m = 1+floor(Ks/2):s2+floor(Ks/2)
            Y(k,m) = sum(sum(A(k-floor(Ks/2):k+floor(Ks/2) , m-floor(Ks/2):m+floor(Ks/2)).*F));
        end
    end
    Y = Y(1+floor(Ks/2):s1+floor(Ks/2),1+floor(Ks/2):s2+floor(Ks/2));
    
    catch
        fprintf('Kernel should be Odd! \n');
    end
end

