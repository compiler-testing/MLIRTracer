module {
  func.func @main(%arg0: tensor<48x7x64x36x59x75xi8>, %arg1: tensor<78xf32>, %arg2: tensor<i1>, %arg3: tensor<i1>) -> (tensor<48x7x64x36x59x75xi8>, tensor<i1>, tensor<78xf32>) {
    %0 = tosa.identity %arg0 : (tensor<48x7x64x36x59x75xi8>) -> tensor<48x7x64x36x59x75xi8>
    %1 = tosa.identity %0 : (tensor<48x7x64x36x59x75xi8>) -> tensor<48x7x64x36x59x75xi8>
    %2 = tosa.minimum %1, %0 : (tensor<48x7x64x36x59x75xi8>, tensor<48x7x64x36x59x75xi8>) -> tensor<48x7x64x36x59x75xi8>
    %3 = tosa.exp %arg1 : (tensor<78xf32>) -> tensor<78xf32>
    %4 = tosa.logical_or %arg2, %arg3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %5 = tosa.pow %3, %3 : (tensor<78xf32>, tensor<78xf32>) -> tensor<78xf32>
    %6 = tosa.pow %5, %3 : (tensor<78xf32>, tensor<78xf32>) -> tensor<78xf32>
    return %2, %4, %6 : tensor<48x7x64x36x59x75xi8>, tensor<i1>, tensor<78xf32>
  }
}
