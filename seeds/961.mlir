module {
  func.func @main(%arg0: tensor<27xi16>, %arg1: tensor<1x2xi64>, %arg2: tensor<5x21x63xi1>, %arg3: tensor<77x77x86x48x34xf32>) -> (tensor<27xi16>, tensor<5x21x63xi1>, tensor<5x21x1xi1>, tensor<77x77x86x48x34xi1>, tensor<77x77x86x48x34xf32>) {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<2xindex>} : () -> !tosa.shape<2>
    %pad_const_0 = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<27xi16>, !tosa.shape<2>, tensor<1xi16>) -> tensor<27xi16>
    %1 = tosa.logical_not %arg2 : (tensor<5x21x63xi1>) -> tensor<5x21x63xi1>
    %2 = tosa.sigmoid %arg3 : (tensor<77x77x86x48x34xf32>) -> tensor<77x77x86x48x34xf32>
    %3 = tosa.logical_and %1, %1 : (tensor<5x21x63xi1>, tensor<5x21x63xi1>) -> tensor<5x21x63xi1>
    %4 = tosa.logical_or %1, %1 : (tensor<5x21x63xi1>, tensor<5x21x63xi1>) -> tensor<5x21x63xi1>
    %5 = tosa.reduce_min %4 {axis = 2 : i32} : (tensor<5x21x63xi1>) -> tensor<5x21x1xi1>
    %6 = tosa.equal %2, %2 : (tensor<77x77x86x48x34xf32>, tensor<77x77x86x48x34xf32>) -> tensor<77x77x86x48x34xi1>
    %7 = tosa.logical_not %6 : (tensor<77x77x86x48x34xi1>) -> tensor<77x77x86x48x34xi1>
    %8 = tosa.pow %2, %2 : (tensor<77x77x86x48x34xf32>, tensor<77x77x86x48x34xf32>) -> tensor<77x77x86x48x34xf32>
    return %0, %3, %5, %7, %8 : tensor<27xi16>, tensor<5x21x63xi1>, tensor<5x21x1xi1>, tensor<77x77x86x48x34xi1>, tensor<77x77x86x48x34xf32>
  }
}
