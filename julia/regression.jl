using Statistics, Distributions


function init_weight(n_features::Int)
    w = rand(Normal(0, 1), n_features)
    return w
end


function regression(x, y, n_iter, λ)
    x = (x .- mean(x, dims=1))./ (std(x, dims=1))
    x = hcat(ones(size(x)[1]), x)
    w = init_weight(size(x)[2])
    error = Any[]
    for i=1:n_iter
        y_pred = x * w
        mse = mean(0.5 * (y - y_pred).^2)
        push!(error, mse)
        grad_w = - transpose(x) * (y - y_pred)
        print(size(grad_w))
        w .-= λ .* grad_w
        println(size(w))
    end
    return error, w 
end


x = rand(Normal(3, 11), (100, 3))
y = rand(Normal(2,5), 100)

regression(x, y, 100, 0.02)
