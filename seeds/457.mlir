module {
  func.func @main(%arg0: tensor<21x18x41x95x55x40xf32>, %arg1: tensor<41x83x9xi32>, %arg2: tensor<41x1x9xi32>) -> (tensor<41x83x1xi32>, tensor<21x18x41x95x55x40xi1>) {
    %0 = tosa.clamp %arg0 {min_val = 3.100000e+01 : f32, max_val = 3.300000e+01 : f32} : (tensor<21x18x41x95x55x40xf32>) -> tensor<21x18x41x95x55x40xf32>
    %1 = tosa.clamp %0 {min_val = 3.100000e+01 : f32, max_val = 3.300000e+01 : f32} : (tensor<21x18x41x95x55x40xf32>) -> tensor<21x18x41x95x55x40xf32>
    %2 = tosa.floor %1 : (tensor<21x18x41x95x55x40xf32>) -> tensor<21x18x41x95x55x40xf32>
    %3 = tosa.rsqrt %2 : (tensor<21x18x41x95x55x40xf32>) -> tensor<21x18x41x95x55x40xf32>
    %4 = tosa.equal %3, %3 : (tensor<21x18x41x95x55x40xf32>, tensor<21x18x41x95x55x40xf32>) -> tensor<21x18x41x95x55x40xi1>
    %5 = tosa.intdiv %arg1, %arg2 : (tensor<41x83x9xi32>, tensor<41x1x9xi32>) -> tensor<41x83x9xi32>
    %6 = tosa.logical_right_shift %5, %5 : (tensor<41x83x9xi32>, tensor<41x83x9xi32>) -> tensor<41x83x9xi32>
    %7 = tosa.abs %4 : (tensor<21x18x41x95x55x40xi1>) -> tensor<21x18x41x95x55x40xi1>
    %8 = tosa.reduce_min %6 {axis = 2 : i32} : (tensor<41x83x9xi32>) -> tensor<41x83x1xi32>
    %9 = tosa.logical_or %7, %4 : (tensor<21x18x41x95x55x40xi1>, tensor<21x18x41x95x55x40xi1>) -> tensor<21x18x41x95x55x40xi1>
    return %8, %9 : tensor<41x83x1xi32>, tensor<21x18x41x95x55x40xi1>
  }
}
