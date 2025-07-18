module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<27x48x55x76x43xi16>, %arg2: tensor<1x1x1x1x43xi16>, %arg3: tensor<8x29x51xi1>, %arg4: tensor<39xi64>, %arg5: tensor<1xi64>) -> (tensor<27x48x55x76x43xi16>, tensor<1xi64>, tensor<f32>, tensor<1x1x51xi1>, tensor<78xi64>, tensor<f32>, tensor<39xi64>, tensor<f32>, tensor<16xi64>, tensor<f32>) {
    %0 = tosa.reciprocal %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.pow %0, %0 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %2 = tosa.bitwise_and %arg1, %arg2 : (tensor<27x48x55x76x43xi16>, tensor<1x1x1x1x43xi16>) -> tensor<27x48x55x76x43xi16>
    %3 = tosa.reduce_all %arg3 {axis = 1 : i32} : (tensor<8x29x51xi1>) -> tensor<8x1x51xi1>
    %4 = tosa.maximum %arg4, %arg5 : (tensor<39xi64>, tensor<1xi64>) -> tensor<39xi64>
    %5 = tosa.ceil %1 : (tensor<f32>) -> tensor<f32>
    %6 = tosa.reduce_sum %4 {axis = 0 : i32} : (tensor<39xi64>) -> tensor<1xi64>
    %7 = tosa.reduce_min %6 {axis = 0 : i32} : (tensor<1xi64>) -> tensor<1xi64>
    %t_8 = tosa.const_shape {values = dense<[ 1 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %8 = tosa.tile %4, %t_8 : (tensor<39xi64>, !tosa.shape<1>) -> tensor<39xi64>
    %9 = tosa.exp %5 : (tensor<f32>) -> tensor<f32>
    %10 = tosa.reduce_any %3 {axis = 0 : i32} : (tensor<8x1x51xi1>) -> tensor<1x1x51xi1>
    %11 = tosa.concat %8, %4 {axis = 0 : i32} : (tensor<39xi64>, tensor<39xi64>) -> tensor<78xi64>
    %12 = tosa.ceil %0 : (tensor<f32>) -> tensor<f32>
    %s_13_start = tosa.const_shape {values = dense<[ 29 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_13_size = tosa.const_shape {values = dense<[ 4 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %13 = tosa.slice %8, %s_13_start, %s_13_size : (tensor<39xi64>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<4xi64>
    %14 = tosa.abs %13 : (tensor<4xi64>) -> tensor<4xi64>
    %15 = tosa.concat %14, %13 {axis = 0 : i32} : (tensor<4xi64>, tensor<4xi64>) -> tensor<8xi64>
    %16 = tosa.bitwise_not %4 : (tensor<39xi64>) -> tensor<39xi64>
    %17 = tosa.pow %1, %5 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %18 = tosa.concat %15, %15 {axis = 0 : i32} : (tensor<8xi64>, tensor<8xi64>) -> tensor<16xi64>
    %19 = tosa.floor %5 : (tensor<f32>) -> tensor<f32>
    return %2, %7, %9, %10, %11, %12, %16, %17, %18, %19 : tensor<27x48x55x76x43xi16>, tensor<1xi64>, tensor<f32>, tensor<1x1x51xi1>, tensor<78xi64>, tensor<f32>, tensor<39xi64>, tensor<f32>, tensor<16xi64>, tensor<f32>
  }
}
