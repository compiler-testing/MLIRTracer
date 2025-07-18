module {
  func.func @main(%arg0: tensor<65x40x46xf32>, %arg1: tensor<65x1x46xf32>, %arg2: tensor<24x71x63x85x87xi32>, %arg3: tensor<79x32x22x26xi1>) -> (tensor<65x40x1xf32>, tensor<79x32x1x26xi1>, tensor<24x71x63x85x87xi1>, tensor<24x71x63x85x87xi32>) {
    %0 = tosa.pow %arg0, %arg1 : (tensor<65x40x46xf32>, tensor<65x1x46xf32>) -> tensor<65x40x46xf32>
    %1 = tosa.clz %arg2 : (tensor<24x71x63x85x87xi32>) -> tensor<24x71x63x85x87xi32>
    %2 = tosa.reduce_all %arg3 {axis = 2 : i32} : (tensor<79x32x22x26xi1>) -> tensor<79x32x1x26xi1>
    %3 = tosa.identity %0 : (tensor<65x40x46xf32>) -> tensor<65x40x46xf32>
    %4 = tosa.clamp %3 {min_val = 9.000000e+00 : f32, max_val = 8.300000e+01 : f32} : (tensor<65x40x46xf32>) -> tensor<65x40x46xf32>
    %5 = tosa.reduce_max %4 {axis = 2 : i32} : (tensor<65x40x46xf32>) -> tensor<65x40x1xf32>
    %6 = tosa.logical_not %2 : (tensor<79x32x1x26xi1>) -> tensor<79x32x1x26xi1>
    %7 = tosa.greater_equal %1, %1 : (tensor<24x71x63x85x87xi32>, tensor<24x71x63x85x87xi32>) -> tensor<24x71x63x85x87xi1>
    %8 = tosa.minimum %1, %1 : (tensor<24x71x63x85x87xi32>, tensor<24x71x63x85x87xi32>) -> tensor<24x71x63x85x87xi32>
    return %5, %6, %7, %8 : tensor<65x40x1xf32>, tensor<79x32x1x26xi1>, tensor<24x71x63x85x87xi1>, tensor<24x71x63x85x87xi32>
  }
}
