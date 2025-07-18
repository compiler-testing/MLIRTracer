module {
  func.func @main(%arg0: tensor<4x10x59x89x57xi32>, %arg1: tensor<1x10x1x1x57xi32>, %arg2: tensor<3x99xi64>, %arg3: tensor<20x3x47x25x31x6xf32>) -> (tensor<11x10x3x9x5xi32>, tensor<4x11xi64>, tensor<20x3x47x25x31x12xf32>, tensor<20x3x94x25x31x12xf32>, tensor<1x2xi64>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<4x10x59x89x57xi32>, tensor<1x10x1x1x57xi32>) -> tensor<4x10x59x89x57xi32>
    %s_1_start = tosa.const_shape {values = dense<[ 0, 0, 4, 4, 1 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %s_1_size = tosa.const_shape {values = dense<[ 11, 10, 3, 9, 5 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %1 = tosa.slice %0, %s_1_start, %s_1_size : (tensor<4x10x59x89x57xi32>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<11x10x3x9x5xi32>
    %2 = tosa.reduce_product %arg2 {axis = 1 : i32} : (tensor<3x99xi64>) -> tensor<3x1xi64>
    %3 = tosa.reduce_product %2 {axis = 0 : i32} : (tensor<3x1xi64>) -> tensor<1x1xi64>
    %4 = tosa.maximum %2, %2 : (tensor<3x1xi64>, tensor<3x1xi64>) -> tensor<3x1xi64>
    %s_5_start = tosa.const_shape {values = dense<[ 0, 0 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_5_size = tosa.const_shape {values = dense<[ 4, 11 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %5 = tosa.slice %4, %s_5_start, %s_5_size : (tensor<3x1xi64>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<4x11xi64>
    %6 = tosa.exp %arg3 : (tensor<20x3x47x25x31x6xf32>) -> tensor<20x3x47x25x31x6xf32>
    %7 = tosa.clamp %6 {min_val = -4.500000e+01 : f32, max_val = 8.900000e+01 : f32} : (tensor<20x3x47x25x31x6xf32>) -> tensor<20x3x47x25x31x6xf32>
    %8 = tosa.concat %7, %7 {axis = 5 : i32} : (tensor<20x3x47x25x31x6xf32>, tensor<20x3x47x25x31x6xf32>) -> tensor<20x3x47x25x31x12xf32>
    %9 = tosa.bitwise_or %3, %3 : (tensor<1x1xi64>, tensor<1x1xi64>) -> tensor<1x1xi64>
    %10 = tosa.add %8, %8 : (tensor<20x3x47x25x31x12xf32>, tensor<20x3x47x25x31x12xf32>) -> tensor<20x3x47x25x31x12xf32>
    %11 = tosa.add %10, %8 : (tensor<20x3x47x25x31x12xf32>, tensor<20x3x47x25x31x12xf32>) -> tensor<20x3x47x25x31x12xf32>
    %12 = tosa.abs %10 : (tensor<20x3x47x25x31x12xf32>) -> tensor<20x3x47x25x31x12xf32>
    %13 = tosa.concat %12, %12 {axis = 2 : i32} : (tensor<20x3x47x25x31x12xf32>, tensor<20x3x47x25x31x12xf32>) -> tensor<20x3x94x25x31x12xf32>
    %14 = tosa.sigmoid %13 : (tensor<20x3x94x25x31x12xf32>) -> tensor<20x3x94x25x31x12xf32>
    %15 = tosa.sigmoid %14 : (tensor<20x3x94x25x31x12xf32>) -> tensor<20x3x94x25x31x12xf32>
    %t_16 = tosa.const_shape {values = dense<[ 3, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %16 = tosa.tile %9, %t_16 : (tensor<1x1xi64>, !tosa.shape<2>) -> tensor<3x2xi64>
    %17 = tosa.reduce_min %16 {axis = 0 : i32} : (tensor<3x2xi64>) -> tensor<1x2xi64>
    %18 = tosa.exp %15 : (tensor<20x3x94x25x31x12xf32>) -> tensor<20x3x94x25x31x12xf32>
    %19 = tosa.reverse %17 {axis = 1 : i32} : (tensor<1x2xi64>) -> tensor<1x2xi64>
    %20 = tosa.minimum %19, %17 : (tensor<1x2xi64>, tensor<1x2xi64>) -> tensor<1x2xi64>
    return %1, %5, %11, %18, %20 : tensor<11x10x3x9x5xi32>, tensor<4x11xi64>, tensor<20x3x47x25x31x12xf32>, tensor<20x3x94x25x31x12xf32>, tensor<1x2xi64>
  }
}
