module {
  func.func @main(%arg0: tensor<41x35x89x3x4x47xi1>, %arg1: tensor<41x35x1x3x1x1xi1>, %arg2: tensor<37x41x100x85x38x82xf32>, %arg3: tensor<91x29xf32>, %arg4: tensor<83x58x65xi32>, %arg5: tensor<83x58x1xi32>) -> (tensor<41x35x89x3x4x47xi1>, tensor<182x29xf32>, tensor<4x1x7x2x7x1xf32>, tensor<83x58x65xi32>, tensor<38x85x41x100x37x82xi1>, tensor<38x85x41x100x37x82xi1>, tensor<37x41x100x85x38x82xf32>, tensor<83x58x1xi32>) {
    %0 = tosa.logical_left_shift %arg0, %arg1 : (tensor<41x35x89x3x4x47xi1>, tensor<41x35x1x3x1x1xi1>) -> tensor<41x35x89x3x4x47xi1>
    %1 = tosa.exp %arg2 : (tensor<37x41x100x85x38x82xf32>) -> tensor<37x41x100x85x38x82xf32>
    %2 = tosa.tanh %1 : (tensor<37x41x100x85x38x82xf32>) -> tensor<37x41x100x85x38x82xf32>
    %3 = tosa.abs %2 : (tensor<37x41x100x85x38x82xf32>) -> tensor<37x41x100x85x38x82xf32>
    %t_4 = tosa.const_shape {values = dense<[ 2, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %4 = tosa.tile %arg3, %t_4 : (tensor<91x29xf32>, !tosa.shape<2>) -> tensor<182x29xf32>
    %5 = tosa.sub %1, %2 : (tensor<37x41x100x85x38x82xf32>, tensor<37x41x100x85x38x82xf32>) -> tensor<37x41x100x85x38x82xf32>
    %6 = tosa.greater %1, %5 : (tensor<37x41x100x85x38x82xf32>, tensor<37x41x100x85x38x82xf32>) -> tensor<37x41x100x85x38x82xi1>
    %7 = "tosa.const"() {values = dense<[4, 3, 1, 2, 0, 5]> : tensor<6xi32>} : () -> tensor<6xi32>
    %8 = tosa.transpose %6 {perms = array<i32: 4, 3, 1, 2, 0, 5>} : (tensor<37x41x100x85x38x82xi1>) -> tensor<38x85x41x100x37x82xi1>
    %9 = tosa.intdiv %arg4, %arg5 : (tensor<83x58x65xi32>, tensor<83x58x1xi32>) -> tensor<83x58x65xi32>
    %10 = tosa.sub %9, %9 : (tensor<83x58x65xi32>, tensor<83x58x65xi32>) -> tensor<83x58x65xi32>
    %s_11_start = tosa.const_shape {values = dense<[ 20, 28, 36, 2, 31, 11 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_11_size = tosa.const_shape {values = dense<[ 4, 1, 7, 2, 7, 1 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %11 = tosa.slice %3, %s_11_start, %s_11_size : (tensor<37x41x100x85x38x82xf32>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<4x1x7x2x7x1xf32>
    %12 = tosa.intdiv %10, %9 : (tensor<83x58x65xi32>, tensor<83x58x65xi32>) -> tensor<83x58x65xi32>
    %13 = tosa.clz %8 : (tensor<38x85x41x100x37x82xi1>) -> tensor<38x85x41x100x37x82xi1>
    %14 = tosa.logical_left_shift %8, %8 : (tensor<38x85x41x100x37x82xi1>, tensor<38x85x41x100x37x82xi1>) -> tensor<38x85x41x100x37x82xi1>
    %15 = tosa.rsqrt %3 : (tensor<37x41x100x85x38x82xf32>) -> tensor<37x41x100x85x38x82xf32>
    %16 = tosa.reduce_sum %10 {axis = 2 : i32} : (tensor<83x58x65xi32>) -> tensor<83x58x1xi32>
    return %0, %4, %11, %12, %13, %14, %15, %16 : tensor<41x35x89x3x4x47xi1>, tensor<182x29xf32>, tensor<4x1x7x2x7x1xf32>, tensor<83x58x65xi32>, tensor<38x85x41x100x37x82xi1>, tensor<38x85x41x100x37x82xi1>, tensor<37x41x100x85x38x82xf32>, tensor<83x58x1xi32>
  }
}
