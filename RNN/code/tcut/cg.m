function [pi,pl,Lq,loglikelihood_samples] = cg( counts, E, W, options )
% [pi,pl,Lq,loglikelihood_samples] =  cg( counts, E, W, options )
% 
% 2-dimensional Counting Grids - UAI 2011
%
% ---- INPUT ----
% counts:            data, features x samples
% E:                 counting grid size  (e.g., [30,30])
% W:                 window size (e.g., [4 4])
% options (Struct) - All fields are optional. 
%   min_change:      convergence criterion. Min relative change in 
%                    loglikelihood (defalut 1e-4)
%   max_iter:        max number of EM iteration
%   learn_pi:        learn counting grid \pi (default 1=true)
%   learn_pl:        learn prior on locations P( \bf{k} ) (default 0=false)
%   plot_figure:     plot the loglikelihood after every iteration
%   normalize_data:  normalize the data (default 1=true, SET TO ZERO FOR
%                    TEXT!)
%   pi:              Learned Counting Grid (for inference, or to pass an
%                    initialization)
%   pl:              Learned Prior (for inference, or to pass an
%                    initialization)
%
% ---- OUTPUT ----
% pi:                counting grid pi(\bf{k},z) = \pi_{\bf{k},z}
% pl:                prior on locations P(i,j)
% Lq:                location posterior for each sample log p(\bf{k}| c^t )
% loglikelihood_samples: loglikelihood of each sample
% 
%
% Written by Alessandro Perina, alessandro.perina@gmail.com /
% alperina@microsoft.com

dbstop if error

if ~exist( 'options', 'var'); options = []; end
if ~isfield( options,'min_change'); options.min_change = 1e-6; end
if ~isfield( options,'max_iter'); options.max_iter = 120; end
if ~isfield( options,'learn_pi'); options.learn_pi = 1; end
if ~isfield( options,'learn_pl'); options.learn_pl = 0; end
if ~isfield( options,'plot_figure'); options.plot_figure = 0; end
if ~isfield( options,'normalize_data'); options.normalize_data = 1; end

[Z,T]=size(counts);
if options.normalize_data
    counts=100*prod(W)*bsxfun( @rdivide, counts, sum(counts,1) );
end
L = prod(E);
total = sum(counts(:));


if isfield( options,'pi')
    pi = options.pi;
else
    pi=1+1*rand([E,Z]);
    pi = bsxfun( @rdivide, pi, sum(pi,3));
end

if isfield( options,'pl'); 
    pl = options.pl;
else
    pl = ones(E)/L;
end

PI = padarray( ...
    permute(cumsum( permute( cumsum( ...
    padarray(pi,W,'circular','post' )),[2 1 3]) ),[2 1 3] ), ...
    [1 1],0,'pre');
tmp = compute_h_noLoopFull( PI, W(2)+1, W(1)+1);
h = bsxfun( @rdivide, tmp(1:end-1,1:end-1,:), sum( tmp(1:end-1,1:end-1,:),3 ));

iter = 1;
alpha = 1e-10;
start_iterating_m = 1; % Start M-step iterations from
m_step_iter = 1; % M-step iterations: fasten convergence

pseudocounts =  mean( sum(counts) / prod(E) )  / 2.5;

converged = iter > options.max_iter;
loglikelihood = zeros(1,options.max_iter );
loglikelihood_samples = zeros(1,T);
minp = 1/(10*L);
Lq = zeros([E,T]);
while ~converged
    
    if options.learn_pl
        lql = bsxfun(@plus, reshape( log( pl ),[L,1]), reshape( log( h), [L,Z])*counts );
    else
        lql = reshape( log( h), [L,Z])*counts;
    end
    Lq = reshape( bsxfun( @minus, bsxfun(@minus, lql, max(lql) ),  log( sum( exp( bsxfun(@minus, lql, max(lql) ) ) ))), [E,T]);
    tmp = exp( Lq ) ;  tmp( tmp< minp ) = minp;   Lq = log( bsxfun(@rdivide, tmp, sum( sum(tmp)) ));
            
    nrm = reshape(  reshape( padarray( exp( Lq), W, 'circular','pre'), [prod(E+W),T])*counts', [ E+W,Z ]);
    
    miter = 1;
    if iter > start_iterating_m
        miter = m_step_iter;
    end
    
    for int_it = 1:miter
        
        if options.learn_pi
            
            QH = permute( cumsum( permute( cumsum(...
                bsxfun( @rdivide, nrm,   padarray(h+prod(W)*alpha ,[W,0],'circular','pre')) ), ...
                [2 1 3]) ),[2 1 3]);
            QH = compute_h_noLoopFull(QH,W(2)+1,W(1)+1);
            QH(QH<0) = 0;
            
            un_pi = pseudocounts  + QH.*(pi+alpha);
            mask = sum(un_pi,3) ~= 0;
            
            pi = bsxfun(@times, bsxfun(@rdivide, un_pi, sum(un_pi,3)), double( mask ) ) ...
                + bsxfun(@times, 1/Z*ones([E,Z]), double( ~mask ) );
            
            PI = padarray( ...
                permute(cumsum( permute( cumsum( ...
                padarray(pi,W,'circular','post' )),[2 1 3]) ),[2 1 3] ), ...
                [1 1],0,'pre');
            tmp = compute_h_noLoopFull( PI, W(2)+1, W(1)+1);
            h = bsxfun( @rdivide, tmp(1:end-1,1:end-1,:), sum( tmp(1:end-1,1:end-1,:),3 ));
        end
        
        if options.learn_pl
            msk = padarray( ones(W),E-W,0,'post');
            pl = zeros(E);
            for t=1:T
                tmp = real( ifft2( fft2( msk ).*fft2( exp(Lq(:,:,t)))) );
                tmp( tmp > 1 ) = 1; tmp( tmp<0) = 0;
                pl = pl + tmp;
            end                
            pl = pl ./ sum(pl(:));
        end
        
    end
    
    loglikelihood_samples = sum( reshape( exp(Lq),[L,T]).*( reshape( log( h), [L,Z])*counts )) - squeeze( sum(sum( exp(Lq).*Lq )) )';
    loglikelihood(iter) = sum( loglikelihood_samples );
    
    if options.plot_figure == 1;
        figure(1), plot(1:iter, loglikelihood(1:iter),'.-r'); grid on; drawnow;
    end
    
    converged = iter >= options.max_iter;
    if iter > 30
        F1 = loglikelihood(iter)/total;
        F2 = loglikelihood(iter-1)/total;
        rel_ch = (F1-F2) / abs(mean([F1,F2]));
        if rel_ch < options.min_change
            converged = 1;
        end
    end
    iter = iter+1;
    
end

end
    
function h = compute_h_noLoopFull( H, xW, yW )
        h = H(yW:end,xW:end,:,:,:) - H(1:end-yW+1,xW:end,:,:,:) ...
            - H(yW:end,1:end-xW+1,:,:,:) + H(1:end-yW+1,1:end-xW+1,:,:,:);    
end