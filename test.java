public class test {
    public static void main(String[] args) {
        // Bug 1: String comparison using ==
        String str = "hello";
        if (str == "hello") {
            System.out.println("String matches");
        }

        // Bug 2: Infinite loop (missing increment)
        int i = 0;
        while (i < 10) {
            System.out.println(i);
            // Missing i++
        }

        // Bug 3: Array index out of bounds potential
        int[] arr = new int[5];
        for (int j = 0; j <= 5; j++) {  // Should be < 5
            arr[j] = j;
        }
    }
} 