#include<bits/stdc++.h>
using namespace std;
// Maximum Score After Splitting a String
class Day_1{
public:
    int maxScore(string s)
    {
        int zeroes = 0, ones = 0, ans = 0;
        for (char i : s)
            if (i == '1')
                ++ones;
        for (int i = 0; i < s.size() - 1; i++)
        {
            if (s[i] == '0')
                zeroes++;
            else
                ones--;
            ans = max(ans, zeroes + ones);
        }
        return ans;
    }
};
//2559. Count Vowel Strings in Ranges
class Day_2{
public:
    vector<int> vowelStrings(vector<string> &words, vector<vector<int>> &queries)
    {
        vector<int> ans;
        unordered_set<char> vowels = {'a', 'e', 'i', 'o', 'u'};
        int sum = 0;
        vector<int> psum(words.size(), 0);
        for (int i = 0; i < words.size(); i++)
        {
            string cur = words[i];
            if (vowels.count(cur[0]) && vowels.count(cur[cur.size() - 1]))
                sum++;
            psum[i] = sum;
        }
        for (auto &i : queries)
        {
            // vector<int> q = queries[i];
            int l = i[0], r = i[1];
            int cur_ans = psum[r] - (l == 0 ? 0 : psum[l - 1]);
            ans.push_back(cur_ans);
        }
        return ans;
    }
};
// 2270. Number of Ways to Split Array
class Day_3{
public:
    int waysToSplitArray(vector<int> &nums)
    {
        long long lsum = 0, rsum = 0;
        for (int i : nums)
            rsum += i;
        int ans = 0;
        for (int i = 0; i < nums.size() - 1; i++)
        {
            lsum += nums[i];
            rsum -= nums[i];

            if (lsum >= rsum)
                ans++;
        }
        return ans;
    }
};
// 1930. Unique Length-3 Palindromic Subsequences
class Day_4{
public:
    int countPalindromicSubsequence(string s)
    {
        unordered_set<char> letters;
        for (auto c : s)
            letters.insert(c);
        int ans = 0;
        for (char ch : letters)
        {
            int i = -1, j = 0;
            for (int k = 0; k < s.size(); k++)
            {
                if (s[k] == ch)
                {
                    if (i == -1)
                        i = k;
                    j = k;
                }
            }
            unordered_set<char> between;
            for (int k = i + 1; k < j; k++)
            {
                between.insert(s[k]);
            }
            ans += between.size();
        }
        return ans;
    }
};
// 2381. Shifting Letters II
class Day_5{
public:
    string shiftingLetters(string s, vector<vector<int>> &shifts)
    {
        int n = s.size();
        vector<int> diff(n + 1, 0);

        for (const auto &shift : shifts)
        {
            int start = shift[0], end = shift[1], direction = shift[2];
            if (direction == 1)
            {
                diff[start] += 1;
                diff[end + 1] -= 1;
            }
            else
            {
                diff[start] -= 1;
                diff[end + 1] += 1;
            }
        }

        int totalShift = 0;
        for (int i = 0; i < n; ++i)
        {
            totalShift += diff[i];
            totalShift = (totalShift % 26 + 26) % 26;
            s[i] = 'a' + (s[i] - 'a' + totalShift) % 26;
        }

        return s;
    }
};
// 1769. Minimum Number of Operations to Move All Balls to Each Box
class Day_6{
public:
    vector<int> minOperations(string boxes)
    {
        vector<int> ans(boxes.size(), 0);
        for (int x = 0; x < boxes.size(); x++)
        {
            if (boxes[x] == '1')
            {
                for (int i = 0; i < boxes.size(); i++)
                {
                    if (i == x)
                        continue;
                    // if(boxes[i] == '1')
                    ans[i] += abs(x - i);
                }
            }
        }
        return ans;
    }
};
// 1408. String Matching in an Array
class Day_7{
public:
    vector<string> stringMatching(vector<string> &words)
    {
        vector<string> ans;
        for (int i = 0; i < words.size(); i++)
        {
            for (int j = 0; j < words.size(); j++)
            {
                if (i != j && words[j].find(words[i]) != string::npos)
                {
                    ans.push_back(words[i]);
                    break;
                }
            }
        }
        return ans;
    }
};
// 3042. Count Prefix and Suffix Pairs I
class Day_8{
public:
    int countPrefixSuffixPairs(vector<string> &words)
    {
        int n = words.size(), count = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                string s1 = words[i], s2 = words[j];
                if (s1.size() > s2.size())
                    continue;
                if (s2.find(s1) == 0 && s2.rfind(s1) == s2.size() - s1.size())
                {
                    count++;
                }
            }
        }
        return count;
    }
};
// 2185. Counting Words With a Given Prefix
class Day_9{
public:
    int prefixCount(vector<string> &words, string pref)
    {
        int count = 0;
        for (auto i : words)
        {
            string s = i;
            string pre = s.substr(0, pref.size());
            if (pre == pref)
                count++;
        }
        return count;
    }
};
// 916. Word Subsets
class Day_10{
private:
    vector<int> count(const string &S)
    {
        vector<int> freq(26, 0);
        for (char c : S)
        {
            freq[c - 'a']++;
        }
        return freq;
    }

public:
    vector<string> wordSubsets(vector<string> &words1, vector<string> &words2)
    {
        vector<int> words2_freq(26, 0); // maximum frequency of words2

        // Compute the maximum frequency of each character across all strings in words2
        for (const string &s : words2)
        {
            vector<int> Count_2 = count(s);
            for (int i = 0; i < 26; ++i)
            {
                words2_freq[i] = max(words2_freq[i], Count_2[i]);
            }
        }

        vector<string> ans;

        // Check each string in A
        for (const string &s : words1)
        {
            vector<int> Count_1 = count(s);
            bool isUniversal = true;

            for (int i = 0; i < 26; ++i)
            {
                if (Count_1[i] < words2_freq[i])
                {
                    isUniversal = false;
                    break;
                }
            }

            if (isUniversal)
            {
                ans.push_back(s);
            }
        }

        return ans;
    }
};
// 1400. Construct K Palindrome Strings
class Day_11{
public:
    bool canConstruct(string s, int k)
    {
        if (s.length() < k)
            return false;
        if (s.length() == k)
            return true;

        vector<int> freq(26);
        int odd = 0; // odd-count
        for (auto &chr : s)
            freq[chr - 'a']++;

        // Count the no. of characters which appear an odd number of times in s
        for (int i = 0; i < 26; i++)
        {
            if (freq[i] % 2 == 1)
            {
                odd++;
            }
        }
        return (odd <= k);
    }
};
// 2116. Check if a Parentheses String Can Be Valid
class Day_12
{
public:
    bool canBeValid(string s, string locked)
    {
        if (s.length() % 2 != 0)
            return false;
        int open = 0, close = 0;
        for (int i = 0; i < s.length(); i++)
        {
            if (locked[i] == '0' || s[i] == '(')
                open++;
            else
                open--;
            if (open < 0)
                return false;
        }
        for (int i = s.length() - 1; i >= 0; i--)
        {
            if (locked[i] == '0' || s[i] == ')')
                close++;
            else
                close--;
            if (close < 0)
                return false;
        }
        return true;
    }
};
// 3223. Minimum Length of String After Operations
class Day_13
{
public:
    int minimumLength(string s)
    {
        unordered_map<char, int> mp;
        for (auto &i : s)
            mp[i]++;
        int del_ct = 0;
        for (auto &i : mp)
        {
            int freq = i.second;
            if (freq % 2 == 1)
            {
                del_ct += freq - 1;
            }
            else
                del_ct += freq - 2;
        }
        return (s.size() - del_ct);
    }
};
// 2657. Find the Prefix Common Array of Two Arrays
class Day_14
{
public:
    vector<int> findThePrefixCommonArray(vector<int> &A, vector<int> &B)
    {
        int n = A.size(), common = 0;
        vector<int> freq(n + 1, 0), ans(n);
        for (int i = 0; i < A.size(); i++)
        {
            if (++freq[A[i]] == 2)
                common++;
            if (++freq[B[i]] == 2)
                common++;
            ans[i] = common;
        }
        return ans;
    }
};
// 2429. Minimize XOR
class Day_15
{
public:
    bool is_set(int x, int bit) { return x & (1 << bit); }
    void set_bit(int &x, int bit) { x |= (1 << bit); }
    void unset(int &x, int bit) { x &= ~(1 << bit); }
    int minimizeXor(int num1, int num2)
    {
        int res = num1, target_set = __builtin_popcount(num2), setbits = __builtin_popcount(res);
        int cur_bit = 0;
        while (setbits < target_set)
        {
            if (!is_set(res, cur_bit))
            {
                set_bit(res, cur_bit);
                setbits++;
            }
            cur_bit++;
        }
        while (setbits > target_set)
        {
            if (is_set(res, cur_bit))
            {
                unset(res, cur_bit);
                setbits--;
            }
            cur_bit++;
        }
        return res;
    }
};
// 2425. Bitwise XOR of All Pairings
class Day_16
{
public:
    int xorAllNums(vector<int> &nums1, vector<int> &nums2)
    {
        int xor1 = 0, xor2 = 0, n1 = nums1.size(), n2 = nums2.size();
        if (n2 % 2 != 0)
        {
            for (int i : nums1)
                xor1 ^= i;
        }
        if (n1 % 2 != 0)
        {
            for (int i : nums2)
                xor2 ^= i;
        }
        return xor1 ^ xor2;
    }
};
// 2683. Neighboring Bitwise XOR
class Day_17
{
public:
    bool doesValidArrayExist(vector<int> &derived)
    {
        vector<int> original = {0};
        for (int i = 0; i < derived.size(); i++)
        {
            original.push_back((derived[i] ^ original[i]));
        }

        bool checkZero = (original[0] == original[original.size() - 1]);
        original = {1};
        for (int i = 0; i < derived.size(); i++)
        {
            original.push_back((derived[i] ^ original[i]));
        }
        bool checkOne = (original[0] == original[original.size() - 1]);

        return checkZero | checkOne;
    }
};
// 1368. Minimum Cost to Make at Least One Valid Path in a Grid
class Day_18
{
public:
    vector<vector<int>> dirs = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

    int minCost(vector<vector<int>> &grid)
    {
        int numRows = grid.size(), numCols = grid[0].size();

        priority_queue<vector<int>, vector<vector<int>>, greater<>> pq;
        pq.push({0, 0, 0});

        vector<vector<int>> minCost(numRows, vector<int>(numCols, INT_MAX));
        minCost[0][0] = 0;

        while (!pq.empty())
        {
            auto curr = pq.top();
            pq.pop();
            int cost = curr[0], row = curr[1], col = curr[2];

            // Skip if we've found a better path to this cell
            if (minCost[row][col] != cost)
                continue;

            // Try all four directions
            for (int dir = 0; dir < 4; dir++)
            {
                int newRow = row + dirs[dir][0];
                int newCol = col + dirs[dir][1];

                // Check if new position is valid
                if (newRow >= 0 && newRow < numRows && newCol >= 0 &&
                    newCol < numCols)
                {
                    // Add cost=1 if we need to change direction
                    int newCost = cost + (dir != (grid[row][col] - 1) ? 1 : 0);

                    // Update if we found a better path
                    if (minCost[newRow][newCol] > newCost)
                    {
                        minCost[newRow][newCol] = newCost;
                        pq.push({newCost, newRow, newCol});
                    }
                }
            }
        }

        return minCost[numRows - 1][numCols - 1];
    }
};
// 407. Trapping Rain Water II
class Day_19
{
public:
    int trapRainWater(vector<vector<int>> &heightMap)
    {
        // Direction arrays
        int dRow[4] = {0, 0, -1, 1};
        int dCol[4] = {-1, 1, 0, 0};

        int numOfRows = heightMap.size();
        int numOfCols = heightMap[0].size();

        vector<vector<bool>> visited(numOfRows, vector<bool>(numOfCols, false));

        // Priority queue (min-heap) to process boundary cells in increasing
        // height order
        priority_queue<Cell> boundary;

        // Add the first and last column cells to the boundary and mark them as
        // visited
        for (int i = 0; i < numOfRows; i++)
        {
            boundary.push(Cell(heightMap[i][0], i, 0));
            boundary.push(Cell(heightMap[i][numOfCols - 1], i, numOfCols - 1));
            // Mark left and right boundary cells as visited
            visited[i][0] = visited[i][numOfCols - 1] = true;
        }

        // Add the first and last row cells to the boundary and mark them as
        // visited
        for (int i = 0; i < numOfCols; i++)
        {
            boundary.push(Cell(heightMap[0][i], 0, i));
            boundary.push(Cell(heightMap[numOfRows - 1][i], numOfRows - 1, i));
            // Mark top and bottom boundary cells as visited
            visited[0][i] = visited[numOfRows - 1][i] = true;
        }

        int totalWaterVolume = 0;

        while (!boundary.empty())
        {
            // Pop the cell with the smallest height from the boundary
            Cell currentCell = boundary.top();
            boundary.pop();

            int currentRow = currentCell.row;
            int currentCol = currentCell.col;
            int minBoundaryHeight = currentCell.height;

            // Explore all 4 neighboring cells
            for (int direction = 0; direction < 4; direction++)
            {
                int neighborRow = currentRow + dRow[direction];
                int neighborCol = currentCol + dCol[direction];

                // Check if the neighbor is within the grid bounds and not yet
                // visited
                if (isValidCell(neighborRow, neighborCol, numOfRows,
                                numOfCols) &&
                    !visited[neighborRow][neighborCol])
                {
                    int neighborHeight = heightMap[neighborRow][neighborCol];

                    // If the neighbor's height is less than the current
                    // boundary height, water can be trapped
                    if (neighborHeight < minBoundaryHeight)
                    {
                        totalWaterVolume += minBoundaryHeight - neighborHeight;
                    }

                    // Push the neighbor into the boundary with updated height
                    // (to prevent water leakage)
                    boundary.push(Cell(max(neighborHeight, minBoundaryHeight),
                                       neighborRow, neighborCol));
                    visited[neighborRow][neighborCol] = true;
                }
            }
        }

        return totalWaterVolume;
    }

private:
    // Struct to store the height and coordinates of a cell in the grid
    class Cell
    {
    public:
        int height;
        int row;
        int col;

        // Constructor to initialize a cell
        Cell(int height, int row, int col)
            : height(height), row(row), col(col) {}

        // Overload the comparison operator to make the priority queue a
        // min-heap based on height
        bool operator<(const Cell &other) const
        {
            // Reverse comparison to simulate a min-heap
            return height >= other.height;
        }
    };

    bool isValidCell(int row, int col, int numOfRows, int numOfCols)
    {
        return row >= 0 && col >= 0 && row < numOfRows && col < numOfCols;
    }
};
// 2661. First Completely Painted Row or Column
class Day_20
{
public:
    int firstCompleteIndex(vector<int> &arr, vector<vector<int>> &mat)
    {
        int numRows = mat.size(), numCols = mat[0].size();
        vector<int> rowCount(numRows), colCount(numCols);
        unordered_map<int, pair<int, int>> numToPos;

        // map to store the position of each number in the matrix
        for (int row = 0; row < numRows; row++)
        {
            for (int col = 0; col < numCols; col++)
            {
                int value = mat[row][col];
                numToPos[value] = {row, col};
            }
        }

        for (int i = 0; i < arr.size(); i++)
        {
            int num = arr[i];
            auto [row, col] = numToPos[num];

            // Increment the count for the corresponding row and column
            rowCount[row]++;
            colCount[col]++;

            // Check if the row or column is completely painted
            if (rowCount[row] == numCols || colCount[col] == numRows)
            {
                return i;
            }
        }

        return -1;
    }
};
// 2017. Grid Game
class Day_21
{
public:
    long long gridGame(vector<vector<int>> &grid)
    {
        long long first_rowsum = accumulate(begin(grid[0]), end(grid[0]), 0LL), sec_rowsum = 0;
        long long minsum = LONG_LONG_MAX;
        // turn index => idx
        for (int idx = 0; idx < grid[0].size(); idx++)
        {
            first_rowsum -= grid[0][idx];
            minsum = min(minsum, max(first_rowsum, sec_rowsum));
            sec_rowsum += grid[1][idx];
        }
        return minsum;
    }
};
// 1765. Map of Highest Peak
class Day_22
{
public:
    vector<vector<int>> highestPeak(vector<vector<int>> &isWater)
    {
        int dx[4] = {0, 0, 1, -1};
        int dy[4] = {1, -1, 0, 0};

        int rows = isWater.size();
        int columns = isWater[0].size();

        vector<vector<int>> cellHeights(rows, vector<int>(columns, -1));

        queue<pair<int, int>> cellQueue;

        for (int x = 0; x < rows; x++)
        {
            for (int y = 0; y < columns; y++)
            {
                if (isWater[x][y])
                {
                    cellQueue.push({x, y});
                    cellHeights[x][y] = 0;
                }
            }
        }

        int heightOfNextLayer = 1;

        while (!cellQueue.empty())
        {
            int layerSize = cellQueue.size();

            for (int i = 0; i < layerSize; i++)
            {
                pair<int, int> currentCell = cellQueue.front();
                cellQueue.pop();

                for (int d = 0; d < 4; d++)
                {
                    pair<int, int> neighborCell = {currentCell.first + dx[d], currentCell.second + dy[d]};

                    if (isValidCell(neighborCell, rows, columns) &&
                        cellHeights[neighborCell.first][neighborCell.second] == -1)
                    {
                        cellHeights[neighborCell.first][neighborCell.second] = heightOfNextLayer;
                        cellQueue.push(neighborCell);
                    }
                }
            }
            heightOfNextLayer++;
        }

        return cellHeights;
    }

private:
    bool isValidCell(pair<int, int> cell, int rows, int columns)
    {
        return cell.first >= 0 && cell.second >= 0 && cell.first < rows &&
               cell.second < columns;
    }
};
// 1267. Count Servers that Communicate
class Day_23
{
public:
    int countServers(vector<vector<int>> &grid)
    {
        vector<int> r_c(grid[0].size(), 0), c_c(grid.size(), 0); // Row - Col Count
        int n = grid.size();
        for (int r = 0; r < n; r++)
            for (int c = 0; c < grid[0].size(); c++)
                if (grid[r][c])
                {
                    r_c[c]++, c_c[r]++;
                }

        int server_c = 0;
        for (int r = 0; r < n; r++)
            for (int c = 0; c < grid[0].size(); c++)
                if (grid[r][c])
                    server_c += r_c[c] > 1 || c_c[r] > 1;
        return server_c;
    }
};
// 802. Find Eventual Safe States
class Day_24
{
public:
    vector<int> eventualSafeNodes(vector<vector<int>> &graph)
    {
        int n = graph.size();
        vector<int> indegree(n);
        vector<vector<int>> adj(n);

        for (int i = 0; i < n; i++)
        {
            for (auto node : graph[i])
            {
                adj[node].push_back(i);
                indegree[i]++;
            }
        }

        queue<int> q;
        for (int i = 0; i < n; i++)
        {
            if (indegree[i] == 0)
            {
                q.push(i);
            }
        }

        vector<bool> safe(n);
        while (!q.empty())
        {
            int node = q.front();
            q.pop();
            safe[node] = true;

            for (auto &neighbor : adj[node])
            {
                // Delete the edge "node -> neighbor".
                indegree[neighbor]--;
                if (indegree[neighbor] == 0)
                {
                    q.push(neighbor);
                }
            }
        }

        vector<int> safeNodes;
        for (int i = 0; i < n; i++)
        {
            if (safe[i])
            {
                safeNodes.push_back(i);
            }
        }
        return safeNodes;
    }
};
// 2948. Make Lexicographically Smallest Array by Swapping Elements
class Day_25
{
public:
    vector<int> lexicographicallySmallestArray(vector<int> &nums, int limit)
    {
        vector<int> sorted(nums);
        sort(sorted.begin(), sorted.end());

        int curr_grp = 0;
        unordered_map<int, int> mp;
        mp.insert(pair<int, int>(sorted[0], curr_grp));

        unordered_map<int, list<int>> grp_lst;
        grp_lst.insert(
            pair<int, list<int>>(curr_grp, list<int>(1, sorted[0])));

        for (int i = 1; i < nums.size(); i++)
        {
            if (abs(sorted[i] - sorted[i - 1]) > limit)
            {

                curr_grp++;
            }

            mp.insert(pair<int, int>(sorted[i], curr_grp));

            if (grp_lst.find(curr_grp) == grp_lst.end())
            {
                grp_lst[curr_grp] = list<int>();
            }
            grp_lst[curr_grp].push_back(sorted[i]);
        }
        for (int i = 0; i < nums.size(); i++)
        {
            int num = nums[i];
            int group = mp[num];
            nums[i] = *grp_lst[group].begin();
            grp_lst[group].pop_front();
        }

        return nums;
    }
};
// 2127. Maximum Employees to Be Invited to a Meeting
class Day_26
{
public:
    int maximumInvitations(vector<int> &favorite)
    {
        int n = favorite.size();
        vector<vector<int>> reversedGraph(n);
        for (int person = 0; person < n; ++person)
        {
            reversedGraph[favorite[person]].push_back(person);
        }

        // Helper function for BFS
        auto bfs = [&](int startNode, unordered_set<int> &visitedNodes) -> int
        {
            queue<pair<int, int>> q;
            q.push({startNode, 0});
            int maxDistance = 0;
            while (!q.empty())
            {
                auto [currentNode, currentDistance] = q.front();
                q.pop();
                for (int neighbor : reversedGraph[currentNode])
                {
                    if (visitedNodes.count(neighbor))
                        continue;
                    visitedNodes.insert(neighbor);
                    q.push({neighbor, currentDistance + 1});
                    maxDistance = max(maxDistance, currentDistance + 1);
                }
            }
            return maxDistance;
        };

        int longestCycle = 0, twoCycleInvitations = 0;
        vector<bool> visited(n, false);

        // Find all cycles
        for (int person = 0; person < n; ++person)
        {
            if (!visited[person])
            {
                unordered_map<int, int> visitedPersons;
                int current = person;
                int distance = 0;
                while (true)
                {
                    if (visited[current])
                        break;
                    visited[current] = true;
                    visitedPersons[current] = distance++;
                    int nextPerson = favorite[current];
                    if (visitedPersons.count(nextPerson))
                    { // Cycle detected
                        int cycleLength = distance - visitedPersons[nextPerson];
                        longestCycle = max(longestCycle, cycleLength);
                        if (cycleLength == 2)
                        {
                            unordered_set<int> visitedNodes = {current,
                                                               nextPerson};
                            twoCycleInvitations +=
                                2 + bfs(nextPerson, visitedNodes) +
                                bfs(current, visitedNodes);
                        }
                        break;
                    }
                    current = nextPerson;
                }
            }
        }

        return max(longestCycle, twoCycleInvitations);
    }
};
// 1462. Course Schedule IV
class Day_27
{
public:
    bool isPrerequisite(unordered_map<int, vector<int>> &adjList,
                        vector<bool> &visited, int src, int target)
    {
        visited[src] = 1;

        if (src == target)
        {
            return true;
        }

        int answer = false;
        for (auto adj : adjList[src])
        {
            if (!visited[adj])
            {
                answer =
                    answer || isPrerequisite(adjList, visited, adj, target);
            }
        }
        return answer;
    }

    vector<bool> checkIfPrerequisite(int numCourses, vector<vector<int>> &prerequisites, vector<vector<int>> &queries)
    {
        unordered_map<int, vector<int>> adjList;
        for (auto edge : prerequisites)
        {
            adjList[edge[0]].push_back(edge[1]);
        }

        vector<bool> answer;
        for (auto q : queries)
        {
            vector<bool> visited(numCourses, false);
            answer.push_back(isPrerequisite(adjList, visited, q[0], q[1]));
        }

        return answer;
    }
};
// 2658. Maximum Number of Fish in a Grid
class Day_28
{
public:
    int cnt_fish(vector<vector<int>> &g, vector<vector<bool>> &vis, int row, int col)
    {
        int r = g.size(), c = g[0].size(), fishes = 0;
        queue<pair<int, int>> q;
        q.push({row, col});
        vis[row][col] = true;

        vector<int> rd = {0, 0, 1, -1}, cd = {1, -1, 0, 0};

        while (!q.empty())
        {
            row = q.front().first;
            col = q.front().second;
            q.pop();
            fishes += g[row][col];
            // Explore all directions
            for (int i = 0; i < 4; i++)
            {
                int nr = row + rd[i], nc = col + cd[i]; // new rows and cols
                if (nr >= 0 && nr < r && nc >= 0 && nc < c && g[nr][nc] && !vis[nr][nc])
                {
                    q.push({nr, nc});
                    vis[nr][nc] = true;
                }
            }
        }
        return fishes;
    }
    int findMaxFish(vector<vector<int>> &grid)
    {
        int r = grid.size(), c = grid[0].size(), res = 0;
        vector<vector<bool>> vis(r, vector<bool>(c));

        for (int i = 0; i < r; i++)
        {
            for (int j = 0; j < c; j++)
            {
                if (grid[i][j] && !vis[i][j])
                {
                    res = max(res, cnt_fish(grid, vis, i, j));
                }
            }
        }
        return res;
    }
};
// 684. Redundant Connection
class Day_29{
private:
    bool isconnected(int src,int target,vector<bool> &vis,vector<int> adj[]){
        vis[src] = true;
        if(src == target) return true;
        int isfound = false;
        for(int a:adj[src]){
            if(!vis[a]){
                isfound = isfound || isconnected(a,target,vis,adj);
            }
        }
        return isfound;
    }
public:
    vector<int> findRedundantConnection(vector<vector<int>> &edges){
        int n = edges.size();
        vector<int> adj[n];
        for(auto e:edges){
            vector<bool> vis(n,false);
            if(isconnected(e[0]-1,e[1]-1,vis,adj))
            return e;
        
        adj[e[0] - 1].push_back(e[1]-1);
        adj[e[1] - 1].push_back(e[0] - 1);
        }
        return {};
    }
};
// 2493. Divide Nodes Into the Maximum Number of Groups
class Day_30{
    public :
        int magnificentSets(int n, vector<vector<int>> &edges){
            vector<vector<int>> adjList(n);
    for (auto edge : edges){
    adjList[edge[0] - 1].push_back(edge[1] - 1);
    adjList[edge[1] - 1].push_back(edge[0] - 1);
    }

        vector<int> colors(n, -1);
        for (int node = 0; node < n; node++)
        {
            if (colors[node] != -1)
                continue;
            colors[node] = 0;
            if (!isBipartite(adjList, node, colors))
                return -1;
        }

        vector<int> distances(n);
        for (int node = 0; node < n; node++)
        {
            distances[node] = getLongestShortestPath(adjList, node, n);
        }

        int maxNumberOfGroups = 0;
        vector<bool> visited(n, false);
        for (int node = 0; node < n; node++)
        {
            if (visited[node])
                continue;
            maxNumberOfGroups += getNumberOfGroupsForComponent(
                adjList, node, distances, visited);
        }

        return maxNumberOfGroups;
    }

private:
    bool isBipartite(vector<vector<int>> &adjList, int node, vector<int> &colors)
    {
        for (int neighbor : adjList[node])
        {
            if (colors[neighbor] == colors[node])
                return false;

            if (colors[neighbor] != -1)
                continue;

            colors[neighbor] = (colors[node] + 1) % 2;

            if (!isBipartite(adjList, neighbor, colors))
                return false;
        }
        return true;
    }

    int getLongestShortestPath(vector<vector<int>> &adjList, int srcNode, int n)
    {
        queue<int> nodesQueue;
        vector<bool> visited(n, false);

        nodesQueue.push(srcNode);
        visited[srcNode] = true;
        int distance = 0;

        while (!nodesQueue.empty())
        {
            int numOfNodesInLayer = nodesQueue.size();
            for (int i = 0; i < numOfNodesInLayer; i++)
            {
                int currentNode = nodesQueue.front();
                nodesQueue.pop();

                for (int neighbor : adjList[currentNode])
                {
                    if (visited[neighbor])
                        continue;
                    visited[neighbor] = true;
                    nodesQueue.push(neighbor);
                }
            }
            distance++;
        }
        return distance;
    }

    int getNumberOfGroupsForComponent(vector<vector<int>> &adjList, int node, vector<int> &distances, vector<bool> &visited)
    {
        int maxNumberOfGroups = distances[node];
        visited[node] = true;

        for (int neighbor : adjList[node])
        {
            if (visited[neighbor])
                continue;
            maxNumberOfGroups = max(maxNumberOfGroups, getNumberOfGroupsForComponent(adjList, neighbor, distances, visited));
        }
        return maxNumberOfGroups;
    }
};
// 827. Making A Large Island
class Day_31
{
private:
    int solve(vector<vector<int>> &grid, int island_id, int row,
              int col)
    {
        if (row < 0 || row >= grid.size() || col < 0 ||
            col >= grid[0].size() ||
            grid[row][col] != 1)
            return 0;

        grid[row][col] = island_id;
        return 1 + solve(grid, island_id, row + 1, col) + solve(grid, island_id, row - 1, col) +
               solve(grid, island_id, row, col + 1) + solve(grid, island_id, row, col - 1);
    }

public:
    int largestIsland(vector<vector<int>> &grid)
    {
        unordered_map<int, int> island_siz;
        int island_id = 2;

        for (int row = 0; row < grid.size(); ++row)
        {
            for (int col = 0; col < grid[0].size();
                 ++col)
            {
                if (grid[row][col] == 1)
                {
                    island_siz[island_id] = solve(
                        grid, island_id, row, col);
                    ++island_id;
                }
            }
        }

        if (island_siz.empty())
        {
            return 1;
        }

        if (island_siz.size() == 1)
        {
            --island_id;
            return (island_siz[island_id] == grid.size() * grid[0].size()) ? island_siz[island_id] : island_siz[island_id] + 1;
        }

        int mx_island_siz = 1;

        for (int row = 0; row < grid.size(); ++row)
        {
            for (int col = 0; col < grid[0].size();
                 ++col)
            {
                if (grid[row][col] == 0)
                {
                    int currentIslandSize = 1;
                    unordered_set<int> neighboringIslands;

                    // Check down
                    if (row + 1 < grid.size() &&
                        grid[row + 1][col] > 1)
                    {
                        neighboringIslands.insert(
                            grid[row + 1][col]);
                    }

                    // Check up
                    if (row - 1 >= 0 &&
                        grid[row - 1][col] > 1)
                    {
                        neighboringIslands.insert(
                            grid[row - 1][col]);
                    }

                    // Check right
                    if (col + 1 < grid[0].size() &&
                        grid[row][col + 1] > 1)
                    {
                        neighboringIslands.insert(
                            grid[row][col + 1]);
                    }

                    // Check left
                    if (col - 1 >= 0 &&
                        grid[row][col - 1] > 1)
                    {
                        neighboringIslands.insert(
                            grid[row][col - 1]);
                    }

                    for (int id : neighboringIslands)
                    {
                        currentIslandSize += island_siz[id];
                    }

                    mx_island_siz = max(mx_island_siz, currentIslandSize);
                }
            }
        }

        return mx_island_siz;
    }
};

int main(){
ios_base::sync_with_stdio(false);
cin.tie(0);
cout.tie(0);
Day_31 sol;
vector<vector<int>> grid = {{1,0},{1,0}};
cout<<sol.largestIsland(grid);
return 0;
}