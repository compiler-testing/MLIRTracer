module {
  func.func @main(%arg0: tensor<99xi1>, %arg1: tensor<99xi1>, %arg2: tensor<92x88x52xf32>) -> (tensor<i1>, tensor<99xi1>, tensor<92x88x52xf32>, tensor<88x52xi32>, tensor<92x88x52xi1>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<99xi1>, tensor<99xi1>) -> tensor<99xi1>
    %1 = tosa.floor %arg2 : (tensor<92x88x52xf32>) -> tensor<92x88x52xf32>
    %2 = tosa.argmax %0 {axis = 0 : i32} : (tensor<99xi1>) -> tensor<i32>
    %3 = tosa.equal %2, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %4 = tosa.logical_and %3, %3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %5 = tosa.log %1 : (tensor<92x88x52xf32>) -> tensor<92x88x52xf32>
    %6 = tosa.argmax %5 {axis = 0 : i32} : (tensor<92x88x52xf32>) -> tensor<88x52xi32>
    %7 = tosa.logical_or %4, %3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %8 = tosa.maximum %6, %6 : (tensor<88x52xi32>, tensor<88x52xi32>) -> tensor<88x52xi32>
    %9 = tosa.arithmetic_right_shift %8, %8 {round = false} : (tensor<88x52xi32>, tensor<88x52xi32>) -> tensor<88x52xi32>
    %10 = tosa.maximum %1, %5 : (tensor<92x88x52xf32>, tensor<92x88x52xf32>) -> tensor<92x88x52xf32>
    %11 = tosa.floor %10 : (tensor<92x88x52xf32>) -> tensor<92x88x52xf32>
    %12 = tosa.logical_not %0 : (tensor<99xi1>) -> tensor<99xi1>
    %13 = tosa.rsqrt %11 : (tensor<92x88x52xf32>) -> tensor<92x88x52xf32>
    %14 = tosa.ceil %13 : (tensor<92x88x52xf32>) -> tensor<92x88x52xf32>
    %15 = tosa.intdiv %9, %6 : (tensor<88x52xi32>, tensor<88x52xi32>) -> tensor<88x52xi32>
    %16 = tosa.greater_equal %10, %13 : (tensor<92x88x52xf32>, tensor<92x88x52xf32>) -> tensor<92x88x52xi1>
    return %7, %12, %14, %15, %16 : tensor<i1>, tensor<99xi1>, tensor<92x88x52xf32>, tensor<88x52xi32>, tensor<92x88x52xi1>
  }
}
